from tradingagents.agents.utils.research_prompts import DEBATE_RULES


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        policy_report = state.get("policy_report", "")
        hot_money_report = state.get("hot_money_report", "")
        lockup_report = state.get("lockup_report", "")

        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Neutral Risk Analyst evaluating an A-share (China mainland) stock, your role is to provide a balanced perspective, weighing both the potential benefits and risks. Factor in A-share market structure, broader trends, and diversification strategies.

A-Share Neutral Framework — use these China-specific balancing considerations:
- T+1 Constraint: Distinguish newly bought shares from existing sellable holdings; do not infer a price direction or prevention of panic selling from this rule
- Policy Sensitivity Calibration: Not all policy signals are equal. Distinguish between top-level State Council directives (high conviction) vs local government incentives (lower reliability) vs market rumors (noise). Weight your risk assessment accordingly.
- Northbound Flow: Verify dates, definitions and scope; do not assume foreign investors are better informed or infer individual-stock buying from market-wide flows
- Valuation Band Approach: Use dated comparable samples and explicit earnings assumptions; if unavailable, do not invent a valuation band or universal anchor
- Lockup Expiry Timing: The neutral view is not to panic at lockup dates but to monitor actual reduction filings (减持公告). The risk is real but the timing is uncertain — reducing exposure gradually near lockup windows is more sensible than binary all-in/all-out.
- Sector Rotation Awareness: Assess dated sector breadth and relative performance; do not assume a fixed rotation duration or a known cycle stage
- Position Sizing over Direction: In a market with ±10-20% daily limits and T+1 settlement, position sizing is more important than directional conviction. A moderate position captures upside while limiting locked-in loss scenarios.

Here is the trader's decision:

{trader_decision}

Challenge both the aggressive and conservative analysts. Point out where each perspective is overly optimistic or overly cautious in the A-share context. Use these data sources:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest News Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Policy Analysis Report: {policy_report}
Hot Money / Capital Flow Report: {hot_money_report}
Lockup Expiry / Insider Reduction Report: {lockup_report}
Data quality assessment and unresolved issues: {state.get('data_quality_summary') or '未提供质量审核结果，不能视为审核通过。'}
Conversation history: {history} Last aggressive argument: {current_aggressive_response} Last conservative argument: {current_conservative_response}. If no responses yet, present your own argument.

Weigh evidence quality rather than averaging opposing opinions; acknowledge when neither case is established.
{DEBATE_RULES}"""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
