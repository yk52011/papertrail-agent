"""Run: streamlit run app.py"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from papertrail.audit import audit_repository, markdown_report
from papertrail.retrieval import RepositoryIndex
from papertrail.store import save_run

load_dotenv()
st.set_page_config(page_title="PaperTrail · 复现诊断", page_icon="🔎", layout="wide")
st.title("PaperTrail")
st.caption("从项目资料中找依据，检查 NLP 实验的复现条件。")

with st.sidebar:
    st.header("项目设置")
    root = st.text_input("本地仓库目录", str(Path("examples/tiny-nlp").resolve()))
    mode = st.radio("模式", ["离线诊断", "模型 Agent"])
    semantic = st.checkbox("启用语义检索", value=False,
                           help="会将索引文件内容发送到配置的 Embedding 服务。")
    st.caption("模型模式会将检索到的资料发送给你配置的模型服务。请先移除私密资料。")
    st.markdown("此页用于资料检索和静态诊断；运行实验和恢复任务请返回实验工作台。")

question = st.text_area("你想检查什么？",
                        "这个项目需要什么数据和环境（dataset / environment / Python）？如何运行最小实验？")
if st.button("开始检查", type="primary"):
    try:
        with st.spinner("正在整理资料和检查依据…"):
            embeddings = None
            if semantic:
                from papertrail.agent import configured_embeddings
                embeddings = configured_embeddings()
                if embeddings is None:
                    raise ValueError("请配置 PAPERTRAIL_EMBEDDING_MODEL 后再启用语义检索。")
            index = RepositoryIndex(Path(root), embeddings=embeddings)
            audit = audit_repository(index)
            answer = ""
            agent_result = None
            hits = index.search(question, mode="hybrid" if embeddings else "bm25")
            if mode == "模型 Agent":
                from papertrail.agent import ask
                agent_result = ask(index, question)
                answer = agent_result["answer"]
                hits = agent_result["evidence"]
            run_id = save_run({"audit": audit, "agent": agent_result})
            st.session_state["result"] = {
                "audit": audit, "answer": answer, "hits": hits, "run_id": run_id,
                "report": markdown_report(audit, question=question, answer=answer),
                "agent": agent_result,
            }
    except Exception as exc:  # noqa: BLE001 -- UI boundary hides provider error payloads.
        # Do not echo provider responses, which may contain sensitive endpoint details.
        st.error(f"检查未完成（{type(exc).__name__}）。请检查路径、配置或服务连接。")

if "result" in st.session_state:
    result = st.session_state["result"]
    audit = result["audit"]
    a, b, c = st.columns(3)
    a.metric("已索引文件", audit["indexed_files"])
    b.metric("证据片段", audit["chunks"])
    c.metric("实验状态", "尚未运行")
    st.info("找到文件只表示存在相关资料；实验运行和论文指标仍待验证。")
    for warning in audit["coverage_warnings"]:
        st.warning(warning)
    tab1, tab2, tab3 = st.tabs(["诊断结果", "原文依据", "执行记录"])
    with tab1:
        if result["answer"]:
            st.markdown(result["answer"])
        st.dataframe([{"检查项": f["check"], "状态": f["status"],
                       "发现文件": ", ".join(f["files"])} for f in audit["findings"]],
                     width="stretch")
        if not result["answer"]:
            st.caption("当前为离线检查：未调用大模型，未生成实验方案。")
    with tab2:
        if not result["hits"]:
            st.write("没有匹配的资料。请尝试资料中使用的英文术语。")
        for hit in result["hits"]:
            with st.expander(hit["citation"]):
                st.code(hit["text"], language="text")
                st.caption("内容指纹：" + hit["digest"][:16])
    with tab3:
        st.write("记录编号：", result["run_id"])
        if result["agent"]:
            st.json({k: v for k, v in result["agent"].items()
                     if k in {"turns", "usage", "budget_exhausted", "retrieval_mode"}})
        st.caption("默认最多 4 次模型调用，每次最多执行 3 个只读工具。")
    st.download_button("下载诊断报告", result["report"], "papertrail-report.md", "text/markdown")
