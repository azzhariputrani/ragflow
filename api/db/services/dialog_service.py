#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
import json
import re
from copy import deepcopy

from api.db import LLMType, ParserType
from api.db.db_models import Dialog, Conversation
from api.db.services.common_service import CommonService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMService, TenantLLMService, LLMBundle
from api.settings import chat_logger, retrievaler, kg_retrievaler
from rag.app.resume import forbidden_select_fields4resume
from rag.nlp import keyword_extraction
from rag.nlp.search import index_name
from rag.utils import rmSpace, num_tokens_from_string, encoder
from api.utils.file_utils import get_project_base_directory


class DialogService(CommonService):
    model = Dialog


class ConversationService(CommonService):
    model = Conversation


def message_fit_in(msg, max_length=4000):
    def count():
        nonlocal msg
        tks_cnts = []
        for m in msg:
            tks_cnts.append(
                {"role": m["role"], "count": num_tokens_from_string(m["content"])})
        total = 0
        for m in tks_cnts:
            total += m["count"]
        return total

    c = count()
    if c < max_length:
        return c, msg

    msg_ = [m for m in msg[:-1] if m["role"] == "system"]
    msg_.append(msg[-1])
    msg = msg_
    c = count()
    if c < max_length:
        return c, msg

    ll = num_tokens_from_string(msg_[0]["content"])
    l = num_tokens_from_string(msg_[-1]["content"])
    if ll / (ll + l) > 0.8:
        m = msg_[0]["content"]
        m = encoder.decode(encoder.encode(m)[:max_length - l])
        msg[0]["content"] = m
        return max_length, msg

    m = msg_[1]["content"]
    m = encoder.decode(encoder.encode(m)[:max_length - l])
    msg[1]["content"] = m
    return max_length, msg


def llm_id2llm_type(llm_id):
    fnm = os.path.join(get_project_base_directory(), "conf")
    llm_factories = json.load(open(os.path.join(fnm, "llm_factories.json"), "r"))
    for llm_factory in llm_factories["factory_llm_infos"]:
        for llm in llm_factory["llm"]:
            if llm_id == llm["llm_name"]:
                return llm["model_type"].strip(",")[-1]


def remove_history_question(messages):
    new_messages = []
    new_messages.append(messages[-1])
    return new_messages


def merge_prompt2question(knowledge, prompt, messages):
    ''''msg[0] knowledge + prompt / msg[1] question'''
    # prompt强制变更为概要生成
    # keywords = ["概要"， "报告"]
    question = messages[-1]
    prompt = "你是一个政务报告总结助手。"
    # prompt_question_1 = ("你是一个文档报告生成助手，按照以下步骤执行：按顺序精简总结每一个段落内容，尤其对于标题内容和数据指标要总结全面。"
    #                      "根据知识库内容只生成2023年相关工作总结")
    prompt_question_1 = ("你是一个文档报告生成助手，按照以下步骤执行：按顺序精简总结每一个段落内容，尤其对于标题内容和数据指标要总结全面。"
                         "根据知识库内容只生成{}".format(question['content']))
    # for k in keywords:
    # if k in question['content']:
    #     prompt = "你是一个文档报告生成助手，按照以下步骤执行：按顺序精简总结每一个段落内容，尤其对于标题内容，要总结全面。"
    #     break
    # 合并 prompt，knowledge，messages
    messages[-1]['content'] = '{} \n 以下是知识库内容：\n {} \n 以上是知识库内容'.format(prompt_question_1, knowledge)
    msg = [{"role": "system", "content": prompt}]
    msg.extend([{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])}
                for m in messages if m["role"] != "system"])
    return msg

def report_summary(dialog, messages, stream=True, **kwargs):
    ans = []
    question = messages[-1]['content']
    messages[-1]['content'] = '{}2023年工作总结'.format(question)
    messages_2023 = deepcopy(messages)
    ans.append(chat(dialog, messages_2023, stream=True, **kwargs))
    messages[-1]['content'] = '{}2024年工作安排'.format(question)
    messages_2024 = deepcopy(messages)
    ans.append(chat(dialog, messages_2024, stream=True, **kwargs))
    messages[-1]['content'] = '{}对省政府工作的意见和建议'.format(question)
    messages_option = deepcopy(messages)
    ans.append(chat(dialog, messages_option, stream=True, **kwargs))
    return ans


def chat(dialog, messages, stream=True, **kwargs):
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."
    llm = LLMService.query(llm_name=dialog.llm_id)
    if not llm:
        llm = TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=dialog.llm_id)
        if not llm:
            raise LookupError("LLM(%s) not found" % dialog.llm_id)
        max_tokens = 200000
    else:
        max_tokens = llm[0].max_tokens
    kbs = KnowledgebaseService.get_by_ids(dialog.kb_ids)
    embd_nms = list(set([kb.embd_id for kb in kbs]))
    if len(embd_nms) != 1:
        yield {"answer": "**ERROR**: Knowledge bases use different embedding models.", "reference": []}
        return {"answer": "**ERROR**: Knowledge bases use different embedding models.", "reference": []}

    is_kg = all([kb.parser_id == ParserType.KG for kb in kbs])
    retr = retrievaler if not is_kg else kg_retrievaler

    # 限制历史问题的数量，只保留最后三个问题
    questions = [m["content"] for m in messages if m["role"] == "user"][-3:]
    attachments = kwargs["doc_ids"].split(",") if "doc_ids" in kwargs else None
    if "doc_ids" in messages[-1]:
        attachments = messages[-1]["doc_ids"]
        for m in messages[:-1]:
            if "doc_ids" in m:
                attachments.extend(m["doc_ids"])
    # embedding 模型切换，目前的bge-embedding模型最大长度为1024，可能限制搜索结果的准确率
    embd_mdl = LLMBundle(dialog.tenant_id, LLMType.EMBEDDING, embd_nms[0])
    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    prompt_config = dialog.prompt_config
    field_map = KnowledgebaseService.get_field_map(dialog.kb_ids)
    # try to use sql if field mapping is good to go
    if field_map:
        chat_logger.info("Use SQL to retrieval:{}".format(questions[-1]))
        ans = use_sql(questions[-1], field_map, dialog.tenant_id, chat_mdl, prompt_config.get("quote", True))
        if ans:
            yield ans
            return

    for p in prompt_config["parameters"]:
        if p["key"] == "knowledge":
            continue
        if p["key"] not in kwargs and not p["optional"]:
            raise KeyError("Miss parameter: " + p["key"])
        if p["key"] not in kwargs:
            prompt_config["system"] = prompt_config["system"].replace(
                "{%s}" % p["key"], " ")

    rerank_mdl = None
    if dialog.rerank_id:
        rerank_mdl = LLMBundle(dialog.tenant_id, LLMType.RERANK, dialog.rerank_id)

    for _ in range(len(questions) // 2):
        questions.append(questions[-1])
    if "knowledge" not in [p["key"] for p in prompt_config["parameters"]]:
        kbinfos = {"total": 0, "chunks": [], "doc_aggs": []}
    else:
        if prompt_config.get("keyword", False):
            questions[-1] += keyword_extraction(chat_mdl, questions[-1])
        kbinfos = retr.retrieval(" ".join(questions), embd_mdl, dialog.tenant_id, dialog.kb_ids, 1, dialog.top_n,
                                 dialog.similarity_threshold,
                                 dialog.vector_similarity_weight,
                                 doc_ids=attachments,
                                 top=dialog.top_k, aggs=False, rerank_mdl=rerank_mdl)

    knowledges = [ck["content_with_weight"] for ck in kbinfos["chunks"]]

    # self-rag
    # if self-rag is enabled or the last knowledge is not relevant to the question, rewrite the question.
    if dialog.prompt_config.get("self_rag") and not relevant(dialog.tenant_id, dialog.llm_id, questions[-1],
                                                             knowledges):
        questions[-1] = rewrite(dialog.tenant_id, dialog.llm_id, questions[-1])
        kbinfos = retr.retrieval(" ".join(questions), embd_mdl, dialog.tenant_id, dialog.kb_ids, 1, dialog.top_n,
                                 dialog.similarity_threshold,
                                 dialog.vector_similarity_weight,
                                 doc_ids=attachments,
                                 top=dialog.top_k, aggs=False, rerank_mdl=rerank_mdl)
        knowledges = [ck["content_with_weight"] for ck in kbinfos["chunks"]]

    chat_logger.info(
        "{}->{}".format(" ".join(questions), "\n->".join(knowledges)))

    # modify by @houzhe
    # 仅选取一个知识库中的文件
    max_simi = 0
    select_idx = -1
    for idx, kb in enumerate(kbinfos["chunks"]):
        if kb["similarity"] >= max_simi:
            max_simi = kb["similarity"]
            select_idx = idx
    kbinfos = {"total": 1, "chunks": [kbinfos['chunks'][select_idx]], "doc_aggs": [kbinfos['doc_aggs'][select_idx]]}
    knowledges = [ck["content_with_weight"] for ck in kbinfos["chunks"]]

    if not knowledges and prompt_config.get("empty_response"):
        yield {"answer": prompt_config["empty_response"], "reference": kbinfos}
        return {"answer": prompt_config["empty_response"], "reference": kbinfos}

    kwargs["knowledge"] = "\n".join(knowledges)
    gen_conf = dialog.llm_setting

    # modify by @houzhe
    # msg = [{"role": "system", "content": prompt_config["system"].format(**kwargs)}]
    # msg.extend([{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])}
    #             for m in messages if m["role"] != "system"])

    msg = merge_prompt2question(knowledges, prompt_config["system"], messages)
    used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.97))

    assert len(msg) >= 2, f"message_fit_in has bug: {msg}"
    if "max_tokens" in gen_conf:
        gen_conf["max_tokens"] = min(
            gen_conf["max_tokens"],
            max_tokens - used_token_count)

    def decorate_answer(answer):
        """
        Decorate the answer with references.
        """
        nonlocal prompt_config, knowledges, kwargs, kbinfos
        refs = []
        if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
            answer, idx = retr.insert_citations(answer,
                                                [ck["content_ltks"]
                                                 for ck in kbinfos["chunks"]],
                                                [ck["vector"]
                                                 for ck in kbinfos["chunks"]],
                                                embd_mdl,
                                                tkweight=1 - dialog.vector_similarity_weight,
                                                vtweight=dialog.vector_similarity_weight)
            idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
            recall_docs = [
                d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
            if not recall_docs: recall_docs = kbinfos["doc_aggs"]
            kbinfos["doc_aggs"] = recall_docs

            refs = deepcopy(kbinfos)
            for c in refs["chunks"]:
                if c.get("vector"):
                    del c["vector"]

        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        return {"answer": answer, "reference": refs}

    if stream:
        answer = ""
        for ans in chat_mdl.chat_streamly(msg[0]["content"], msg[1:], gen_conf):
            answer = ans
            yield {"answer": answer, "reference": {}}
        yield decorate_answer(answer)
    else:
        answer = chat_mdl.chat(
            msg[0]["content"], msg[1:], gen_conf)
        chat_logger.info("User: {}|Assistant: {}".format(
            msg[-1]["content"], answer))
        yield decorate_answer(answer)


def use_sql(question, field_map, tenant_id, chat_mdl, quota=True):
    """
    Generate and execute an SQL query based on a user question and a field map.

    Args:
        question (str): The user's question.
        field_map (dict): A dictionary mapping field names to their descriptions.
        tenant_id (str): The tenant identifier.
        chat_mdl (LLMBundle): The language model bundle used for generating SQL.
        quota (bool, optional): Whether to include quota information in the results. Defaults to True.

    Returns:
        dict: A dictionary containing the SQL query results and references.
    """
    sys_prompt = "你是一个DBA。你需要这对以下表的字段结构，根据用户的问题列表，写出最后一个问题对应的SQL。"
    user_promt = """
表名：{}；
数据库表字段说明如下：
{}

问题如下：
{}
请写出SQL, 且只要SQL，不要有其他说明及文字。
""".format(
        index_name(tenant_id),
        "\n".join([f"{k}: {v}" for k, v in field_map.items()]),
        question
    )
    tried_times = 0

    def get_table():
        nonlocal sys_prompt, user_promt, question, tried_times
        sql = chat_mdl.chat(sys_prompt, [{"role": "user", "content": user_promt}], {
            "temperature": 0.06})
        print(user_promt, sql)
        chat_logger.info(f"“{question}”==>{user_promt} get SQL: {sql}")
        sql = re.sub(r"[\r\n]+", " ", sql.lower())
        sql = re.sub(r".*select ", "select ", sql.lower())
        sql = re.sub(r" +", " ", sql)
        sql = re.sub(r"([;；]|```).*", "", sql)
        if sql[:len("select ")] != "select ":
            return None, None
        if not re.search(r"((sum|avg|max|min)\(|group by )", sql.lower()):
            if sql[:len("select *")] != "select *":
                sql = "select doc_id,docnm_kwd," + sql[6:]
            else:
                flds = []
                for k in field_map.keys():
                    if k in forbidden_select_fields4resume:
                        continue
                    if len(flds) > 11:
                        break
                    flds.append(k)
                sql = "select doc_id,docnm_kwd," + ",".join(flds) + sql[8:]

        print(f"“{question}” get SQL(refined): {sql}")

        chat_logger.info(f"“{question}” get SQL(refined): {sql}")
        tried_times += 1
        return retrievaler.sql_retrieval(sql, format="json"), sql

    tbl, sql = get_table()
    if tbl is None:
        return None
    if tbl.get("error") and tried_times <= 2:
        user_promt = """
        表名：{}；
        数据库表字段说明如下：
        {}

        问题如下：
        {}

        你上一次给出的错误SQL如下：
        {}

        后台报错如下：
        {}

        请纠正SQL中的错误再写一遍，且只要SQL，不要有其他说明及文字。
        """.format(
            index_name(tenant_id),
            "\n".join([f"{k}: {v}" for k, v in field_map.items()]),
            question, sql, tbl["error"]
        )
        tbl, sql = get_table()
        chat_logger.info("TRY it again: {}".format(sql))

    chat_logger.info("GET table: {}".format(tbl))
    print(tbl)
    if tbl.get("error") or len(tbl["rows"]) == 0:
        return None

    docid_idx = set([ii for ii, c in enumerate(
        tbl["columns"]) if c["name"] == "doc_id"])
    docnm_idx = set([ii for ii, c in enumerate(
        tbl["columns"]) if c["name"] == "docnm_kwd"])
    clmn_idx = [ii for ii in range(
        len(tbl["columns"])) if ii not in (docid_idx | docnm_idx)]

    # compose markdown table
    clmns = "|" + "|".join([re.sub(r"(/.*|（[^（）]+）)", "", field_map.get(tbl["columns"][i]["name"],
                                                                        tbl["columns"][i]["name"])) for i in
                            clmn_idx]) + ("|Source|" if docid_idx and docid_idx else "|")

    line = "|" + "|".join(["------" for _ in range(len(clmn_idx))]) + \
           ("|------|" if docid_idx and docid_idx else "")

    rows = ["|" +
            "|".join([rmSpace(str(r[i])) for i in clmn_idx]).replace("None", " ") +
            "|" for r in tbl["rows"]]
    if quota:
        rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
    else:
        rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
    rows = re.sub(r"T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+Z)?\|", "|", rows)

    if not docid_idx or not docnm_idx:
        chat_logger.warning("SQL missing field: " + sql)
        return {
            "answer": "\n".join([clmns, line, rows]),
            "reference": {"chunks": [], "doc_aggs": []}
        }

    docid_idx = list(docid_idx)[0]
    docnm_idx = list(docnm_idx)[0]
    doc_aggs = {}
    for r in tbl["rows"]:
        if r[docid_idx] not in doc_aggs:
            doc_aggs[r[docid_idx]] = {"doc_name": r[docnm_idx], "count": 0}
        doc_aggs[r[docid_idx]]["count"] += 1
    return {
        "answer": "\n".join([clmns, line, rows]),
        "reference": {"chunks": [{"doc_id": r[docid_idx], "docnm_kwd": r[docnm_idx]} for r in tbl["rows"]],
                      "doc_aggs": [{"doc_id": did, "doc_name": d["doc_name"], "count": d["count"]} for did, d in
                                   doc_aggs.items()]}
    }


def relevant(tenant_id, llm_id, question, contents: list):
    """
    Check if the retrieved documents are relevant to the user's question by using a language model to grade the relevance.

    Args:
        tenant_id (str): The tenant identifier.
        llm_id (str): The language model identifier.
        question (str): The user's question.
        contents (list): A list of document contents to be checked for relevance.

    Returns:
        bool: True if the documents are relevant, False otherwise.
    """
    if llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
    # prompt = """
    #     You are an expert at query expansion to generate a paraphrasing of a question.
    #     I can't retrieval relevant information from the knowledge base by using user's question directly.
    #     You need to expand or paraphrase user's question by multiple ways such as using synonyms words/phrase,
    #     writing the abbreviation in its entirety, adding some extra descriptions or explanations,
    #     changing the way of expression, translating the original question into another language (English/Chinese), etc.
    #     And return 5 versions of question and one is from translation.
    #     Just list the question. No other words are needed.
    # """
    prompt = """
        You are a grader assessing relevance of a retrieved document to a user question. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        No other words needed except 'yes' or 'no'.
    """
    if not contents: return False
    contents = "Documents: \n" + "   - ".join(contents)
    contents = f"Question: {question}\n" + contents
    if num_tokens_from_string(contents) >= chat_mdl.max_length - 4:
        contents = encoder.decode(encoder.encode(contents)[:chat_mdl.max_length - 4])
    ans = chat_mdl.chat(prompt, [{"role": "user", "content": contents}], {"temperature": 0.01})
    if ans.lower().find("yes") >= 0: return True
    return False


def rewrite(tenant_id, llm_id, question):
    """
    Generate multiple paraphrased versions of a user's question to improve retrieval accuracy.

    Args:
        tenant_id (str): The tenant identifier.
        llm_id (str): The language model identifier.
        question (str): The user's original question.

    Returns:
        str: A string containing multiple paraphrased versions of the question.
    """
    if llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
    prompt = """
        你是一个问题重述专家。你需要对用户的问题进行重述，以提高检索的准确性。你需要通过多种方式对用户的问题进行扩展或改写，例如使用同义词/短语、将缩写写成全称、添加额外的描述或解释、改变表达方式、将原问题翻译成另一种语言（英文/中文）等。并返回5个问题版本，其中一个是翻译版本。
        只需列出问题。不需要其他文字。
    """
    ans = chat_mdl.chat(prompt, [{"role": "user", "content": question}], {"temperature": 0.8})
    return ans
