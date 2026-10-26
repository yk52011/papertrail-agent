"""PaperTrail workbench. Start with: streamlit run app.py."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from papertrail.jobs import JobStore
from papertrail.workflow import job_summary, resume_job, start_job

PROJECT = Path(__file__).resolve().parent
load_dotenv(PROJECT / '.env')
st.set_page_config(page_title='PaperTrail · 实验工作台', page_icon='◈', layout='wide')
st.markdown('''<style>
@media (max-width:760px) {
 .st-key-summary [data-testid="stHorizontalBlock"] {flex-wrap:nowrap!important;gap:0.6rem}
 .st-key-summary [data-testid="stColumn"] {min-width:0!important;flex:1!important}
 .st-key-summary [data-testid="stMetricValue"] {font-size:1.35rem}
}
</style>''', unsafe_allow_html=True)

STATUS = {'created': '已创建', 'running': '执行中', 'recovering': '恢复中', 'paused': '等待继续',
          'succeeded': '核验通过', 'failed': '未完成', 'cancelled': '已停止',
          'budget_exhausted': '预算用完'}
POLICIES = {'规则恢复 · 无需密钥': 'rules', '模型决策 · LangChain': 'llm',
            '仅执行一次 · 对照': 'none', '原样重试 · 对照': 'retry'}


def load_selected(store):
    job_id = st.session_state.get('selected_job_id')
    return store.load(job_id) if job_id else None


def task_report(state):
    lines = ['# PaperTrail 实验执行报告', '', f"任务：{state['task']['title']}",
             f"状态：{STATUS.get(state['status'], state['status'])}",
             f"决策策略：{state['policy']}", '', '## 执行与核验', '']
    for attempt in state.get('attempts', []):
        check = attempt['verification']
        lines += [(f"- 第 {attempt['attempt']} 次：退出码 {attempt['exit_code']}；核验 {check['status']}；"
                   f"耗时 {attempt['seconds']} 秒。")]
    lines += ['', '## 配置修改', '']
    for patch in state.get('patches', []):
        for change in patch['changes']:
            lines += [f"- {patch['file']} / {change['key']}: {change['before']} → {change['after']}"]
        lines += ['  依据：' + ', '.join(patch['citations'])]
    lines += ['', '## 边界', '', '仅核验当前清单和数据。未验证任何论文指标；规则模式未调用大模型。']
    return '\n'.join(lines)


def render_attempts(state):
    for attempt in state.get('attempts', []):
        check = attempt['verification']
        with st.expander(f"第 {attempt['attempt']} 次执行 · {check['status']} · {attempt['seconds']} 秒",
                         expanded=not check['verified']):
            st.code(' '.join(attempt['command']), language='bash')
            st.code(attempt['output'] or '（没有标准输出）', language='text')
            if check.get('independent'):
                metrics = check['independent']
                left, right = st.columns(2)
                left.metric('独立重算 Accuracy', f"{metrics['accuracy']:.3f}")
                right.metric('独立重算 Macro F1', f"{metrics['macro_f1']:.3f}")
                st.caption(f"覆盖 {metrics['count']} 个样本；错误 ID：{metrics['wrong_ids'] or '无'}")
            if check.get('artifact', {}).get('predictions'):
                st.dataframe(check['artifact']['predictions'], width='stretch', hide_index=True)


def workbench():
    store = JobStore(PROJECT / '.papertrail/jobs')
    st.caption('PAPERTRAIL / EXPERIMENT WORKBENCH')
    st.title('PaperTrail 实验工作台')
    st.write('执行实验、定位失败、检索依据、修复配置，再独立核验结果。')

    with st.sidebar:
        st.header('任务设置')
        source = st.text_input('实验仓库', str(PROJECT / 'examples/sentiment-lab'))
        policy_label = st.selectbox('决策方式', list(POLICIES))
        backend_label = st.selectbox('运行环境', ['本机 · 仅可信代码', 'Docker · 禁网执行'])
        max_attempts = st.slider('最多执行次数', 1, 6, 3)
        max_seconds = st.slider('任务时间预算 / 秒', 30, 300, 120, step=30)
        step_mode = st.toggle('逐步演示：修复前暂停', value=False)
        st.caption('本机运行没有操作系统隔离。默认仓库为本项目自带的可读演示代码。')
        st.divider()
        st.caption('模型模式从本地 .env 读取配置；规则模式不调用模型。')

    with st.container(border=True):
        st.caption('检查项目 → 执行实验 → 核验产物 → 检索与诊断 → 修复重跑')

    current = load_selected(store)
    summary = current['summary'] if current else {}
    with st.container(key='summary'):
        cols = st.columns(4)
        cols[0].metric('当前状态', STATUS.get(current['status'], current['status']) if current else '等待开始')
        cols[1].metric('执行次数', summary.get('attempts', 0))
        cols[2].metric('配置修复', summary.get('repairs', 0))
        cols[3].metric('模型调用', summary.get('model_calls', 0))

    task_tab, trace_tab, eval_tab, about_tab = st.tabs(['实验任务', '过程与证据', '对照评测', '工程说明'])
    with task_tab:
        main, history = st.columns([2, 1])
        with main:
            with st.container(border=True):
                st.subheader('一个可重复演示的失败恢复案例')
                st.write('Sentiment Lab 使用 TF-IDF + 逻辑回归，在 CPU 上训练文本分类器。'
                         '初始配置故意包含错误的数据路径和文本字段，需要两次恢复才能继续。')
                st.caption('数据为 48 条人工训练句子和 20 条人工测试句子；用于演示工程流程。')
                if st.button('启动实验任务', type='primary', width='stretch'):
                    try:
                        with st.spinner('正在执行任务；结果与每一步状态会写入本地检查点…'):
                            result = start_job(Path(source), store=store,
                                backend='local' if backend_label.startswith('本机') else 'docker',
                                policy=POLICIES[policy_label], max_attempts=max_attempts,
                                max_seconds=max_seconds, step_mode=step_mode)
                        st.session_state['selected_job_id'] = result['job_id']
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001 -- UI boundary avoids provider details.
                        st.error(f'任务未完成（{type(exc).__name__}），请检查配置与服务连接。')
            if current and current['state']:
                state = current['state']
                st.subheader(STATUS.get(state['status'], state['status']))
                if state['status'] == 'paused':
                    st.info('任务已保存检查点。可以先查看建议和依据，关闭页面后仍可从历史记录继续。')
                    st.json(state.get('decision', {}))
                    go, cancel = st.columns(2)
                    if go.button('继续这次修复', type='primary'):
                        try:
                            with st.spinner('从磁盘检查点继续…'):
                                resume_job(state['job_id'], store=store)
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001 -- same UI boundary.
                            st.error(f'继续任务失败（{type(exc).__name__}）。')
                    if cancel.button('停止任务'):
                        resume_job(state['job_id'], proceed=False, store=store)
                        st.rerun()
                elif state['status'] == 'succeeded':
                    st.success('新生成的预测产物通过核验。已检查样本覆盖，并独立重算指标。'
                               if summary.get('verification', {}).get('independent') else
                               '产物通过清单核验；此任务未配置独立标签核验。')
                else:
                    st.warning(state.get('stop_reason') or '任务未达到产物条件或已用完预算。')
                render_attempts(state)
                st.download_button('下载任务报告', task_report(state), 'papertrail-task.md', 'text/markdown')
                st.download_button('下载完整执行记录', json.dumps(state, ensure_ascii=False, indent=2),
                                   'papertrail-trace.json', 'application/json')
        with history:
            st.subheader('任务历史')
            jobs = store.list(20)
            if not jobs:
                st.caption('启动任务后，这里会保留结果和可继续的检查点。')
            for job in jobs:
                with st.container(border=True):
                    st.write(job['title'])
                    st.caption(f"{STATUS.get(job['status'], job['status'])} · {job['id'][:8]}")
                    if st.button('查看任务', key=job['id'], width='stretch'):
                        st.session_state['selected_job_id'] = job['id']
                        st.rerun()
    with trace_tab:
        if not current or not current['state']:
            st.info('启动或选择一个任务，查看实际发生的步骤。')
        else:
            state = current['state']
            timeline, evidence = st.columns([3, 2])
            with timeline:
                st.subheader('执行时间线')
                for index, item in enumerate(state.get('events', []), 1):
                    with st.expander(f"{index:02} · {item['node']} · {item['summary']}"):
                        st.json(item)
                st.subheader('配置变化')
                for patch in state.get('patches', []):
                    st.dataframe(patch['changes'], width='stretch', hide_index=True)
                    st.caption('文件：' + patch['file'] + ' · 依据：' + ', '.join(patch['citations']))
            with evidence:
                st.subheader('最近一次诊断依据')
                for hit in state.get('evidence', []):
                    with st.expander(hit['citation']):
                        st.code(hit['text'], language='text')
                st.caption('引用位置已校验属于当前检索结果；尚未自动判断原文是否充分支持每个决策。')
                st.subheader('资源记录')
                st.json({'policy': state['policy'], 'model_calls': state['model_calls'],
                         'usage': state.get('usage', []), 'summary': job_summary(state)})
    with eval_tab:
        st.subheader('恢复是否有效，用相同任务来比较')
        st.caption('10 个明确注入的故障案例；单次执行、原样重试和规则恢复使用相同代码、数据与阈值。'
                   '这不是真实仓库成功率，也不是模型能力评测。')
        result_path = PROJECT / 'eval/results/recovery.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            table = [{'策略': row['policy'], '通过任务': f"{row['successes']}/{row['total']}",
                      '误报成功': row['false_successes'], '总执行次数': row['total_executions'],
                      '净改善任务数': row['net_gain_count']} for row in result['summaries']]
            st.dataframe(table, width='stretch', hide_index=True)
            st.bar_chart({row['policy']: row['successes'] for row in result['summaries']})
            with st.expander('逐个案例：修改了什么，哪些任务改善了？'):
                st.dataframe([{k: v for k, v in row.items() if k not in {'events', 'changes'}}
                              for row in result['samples']], width='stretch', hide_index=True)
                st.json(result['summaries'])
            st.download_button('下载逐案例评测结果', result_path.read_text(), 'recovery-results.json')
        else:
            st.info('尚无恢复评测结果。运行 scripts/evaluate_recovery.py 后显示真实数据。')
    with about_tab:
        st.subheader('技术与实际职责')
        st.table([
            {'技术': 'LangGraph + SQLiteSaver', '用途': '执行分支、恢复循环、逐节点持久化、暂停与继续'},
            {'技术': 'LangChain + Pydantic', '用途': '模型调用与结构化恢复决策，校验配置修改权限'},
            {'技术': 'RAG', '用途': '根据失败检索带文件行号的项目依据；支持可选语义融合检索'},
            {'技术': '独立核验器', '用途': '检查新产物、预测 ID 覆盖、重算 Accuracy 与 Macro F1'},
            {'技术': '受限执行器', '用途': '临时副本、进程超时、日志上限、可选 Docker 禁网环境'},
            {'技术': '控制实验', '用途': '固定故障集与基线，记录改善、损坏和误报成功的案例'},
        ])
        st.write('已接入实际执行与配置恢复，不自动安装依赖或修改任意代码。'
                 '规则演示与模型决策分别标记；论文指标始终未验证。')
        st.caption('模型 API 与 Docker 尚未在当前环境进行真实调用；对应路径与测试替身已实现。')
        st.code('uv run papertrail reproduce examples/sentiment-lab --backend local --policy rules\n'
                'uv run papertrail reproduce examples/sentiment-lab --backend local --step\n'
                'uv run papertrail resume <job_id>', language='bash')


navigation = st.navigation([
    st.Page(workbench, title='实验工作台', icon=':material/science:', default=True),
    st.Page('pages/2_资料诊断.py', title='资料检索与诊断', icon=':material/manage_search:'),
])
navigation.run()
