import re

with open("streamlit_app.py", "r", encoding="utf-8") as f:
    content = f.read()

target = """        res = st.session_state.room4_result
        if res:
            df = pd.DataFrame([asdict(m) for m in res.metrics])
            summary_cols = st.columns(3)
            summary_cols[0].metric(
                "Training duration",
                _format_training_duration(
                    float(getattr(res, "training_duration_seconds", 0.0))
                ),
            )
            summary_cols[1].metric("Episodes completed", res.episodes_run)
            summary_cols[2].metric(
                "Actions selected", sum(getattr(res, "action_counts", {}).values())
            )

            if not requests["train"]:
                st.subheader("Training metrics")
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("**Total reward per episode**")
                    render_locked_line_chart(df, x="episode", y="total_reward", x_label="Episode", y_label="Total reward")
                    st.caption("**Entropy**")
                    if "entropy" in df.columns:
                        render_locked_line_chart(df, x="episode", y="entropy", x_label="Episode", y_label="Entropy")
                with c2:
                    st.caption("**Policy loss**")
                    if "policy_loss" in df.columns:
                        render_locked_line_chart(df, x="episode", y="policy_loss", x_label="Episode", y_label="Policy loss")
                    st.caption("**Value loss**")
                    if "value_loss" in df.columns:
                        render_locked_line_chart(df, x="episode", y="value_loss", x_label="Episode", y_label="Value loss")

            st.subheader("Training action distribution")
            render_locked_bar_chart(
                _room4_action_dataframe(getattr(res, "action_counts", {})),
                x="Action",
                y="Selections",
                x_label="Action",
                y_label="Number of selections",
            )

            st.write(f"**Episodes run:** {res.episodes_run} | **Goal reached in late training:** {'Yes ?' if res.converged else 'No ?'}")
            if hasattr(res, "training_episodes") and res.training_episodes:
                env_cur = st.session_state.room4_result_environment or build_room4_environment()
                render_episode_replay_visualizer(
                    env_cur,
                    res.training_episodes,
                    "room4_tr_replay",
                    4,
                    title="Training Episodes Replay",
                )
        else:"""

replacement = """        res = st.session_state.room4_result
        if res:
            _render_room4_training_summary(res, show_charts=not requests["train"])
        else:"""

new_func = """def _render_room4_training_summary(result: Any, *, show_charts: bool = True) -> None:
    df = pd.DataFrame([asdict(m) for m in result.metrics])
    summary_cols = st.columns(3)
    summary_cols[0].metric("Training duration", _format_training_duration(float(getattr(result, "training_duration_seconds", 0.0))))
    summary_cols[1].metric("Episodes completed", result.episodes_run)
    summary_cols[2].metric("Actions selected", sum(getattr(result, "action_counts", {}).values()))

    if show_charts:
        st.subheader("Training metrics")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("**Total reward per episode**")
            render_locked_line_chart(df, x="episode", y="total_reward", x_label="Episode", y_label="Total reward")
            st.caption("**Entropy**")
            if "entropy" in df.columns:
                render_locked_line_chart(df, x="episode", y="entropy", x_label="Episode", y_label="Entropy")
        with c2:
            st.caption("**Policy loss**")
            if "policy_loss" in df.columns:
                render_locked_line_chart(df, x="episode", y="policy_loss", x_label="Episode", y_label="Policy loss")
            st.caption("**Value loss**")
            if "value_loss" in df.columns:
                render_locked_line_chart(df, x="episode", y="value_loss", x_label="Episode", y_label="Value loss")

    st.subheader("Training action distribution")
    render_locked_bar_chart(
        _room4_action_dataframe(getattr(result, "action_counts", {})),
        x="Action", y="Selections", x_label="Action", y_label="Number of selections"
    )

    st.write(f"**Episodes run:** {result.episodes_run} | **Goal reached in late training:** {'Yes ?' if result.converged else 'No ?'}")
    if hasattr(result, "training_episodes") and result.training_episodes:
        env_cur = st.session_state.room4_result_environment or build_room4_environment()
        render_episode_replay_visualizer(
            env_cur, result.training_episodes, "room4_tr_replay", 4, title="Training Episodes Replay"
        )

def _render_room4_section("""

if target in content:
    content = content.replace(target, replacement)
    content = content.replace("def _render_room4_section(", new_func)
    with open("streamlit_app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("done")
else:
    print("Target not found")
