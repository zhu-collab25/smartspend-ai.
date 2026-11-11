
import streamlit as st
import pandas as pd
from pathlib import Path
from joblib import load

# ---------------------------------
# 1. PAGE SETUP
# ---------------------------------

st.set_page_config(
    page_title="SmartSpend AI",
    page_icon="💰",
    layout="wide"
)

st.title("💰 SmartSpend AI")
st.caption("Your Intelligent Personal Budgeting Assistant")

st.markdown(
    "Understand your spending, manage your budget, "
    "and work towards your savings goals."
)

# ---------------------------------
# 2. LOAD AI MODEL
# ---------------------------------

model_path = Path(__file__).resolve().parent / "classifier.joblib"

@st.cache_resource
def load_model():
    return load(model_path)

model = load_model()

categories = [
    "Food & Drinks",
    "Transportation",
    "Shopping",
    "Entertainment",
    "Groceries"
]

# ---------------------------------
# 3. SIDEBAR
# ---------------------------------

st.sidebar.title("⚙️ Financial Settings")

income = st.sidebar.number_input(
    "Monthly Income ($)",
    min_value=0.0,
    value=600.0,
    step=50.0
)

budget = st.sidebar.number_input(
    "Monthly Spending Budget ($)",
    min_value=0.0,
    value=500.0,
    step=50.0
)

savings_goal = st.sidebar.number_input(
    "Monthly Savings Goal ($)",
    min_value=0.0,
    value=100.0,
    step=25.0
)

st.sidebar.divider()

st.sidebar.info(
    "SmartSpend uses machine learning to "
    "predict expense categories."
)

# ---------------------------------
# 4. UPLOAD TRANSACTIONS
# ---------------------------------

st.header("📁 Upload Your Transactions")

uploaded_file = st.file_uploader(
    "Upload a CSV file with Description and Amount columns",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    required = {"Description", "Amount"}

    if not required.issubset(df.columns):
        st.error(
            "Your CSV must contain Description and Amount columns."
        )
        st.stop()

    df["Description"] = df["Description"].fillna("").astype(str)

    df["Amount"] = pd.to_numeric(
        df["Amount"],
        errors="coerce"
    )

    if df["Amount"].isna().any():
        st.error("Some transaction amounts are invalid.")
        st.stop()

    if (df["Amount"] < 0).any():
        st.error(
            "Please use positive amounts for expenses."
        )
        st.stop()

    if df.empty:
        st.warning("Your CSV file contains no transactions.")
        st.stop()

    # ---------------------------------
    # 5. AI PREDICTIONS
    # ---------------------------------

    # Keep user corrections when Streamlit reruns
    file_signature = (
        uploaded_file.name,
        uploaded_file.size,
        uploaded_file.getvalue()
    )

    if st.session_state.get("file_signature") != file_signature:

        df["Category"] = model.predict(
            df["Description"]
        )

        st.session_state["transactions"] = df.copy()
        st.session_state["file_signature"] = file_signature

    st.success("AI has categorized your transactions!")

    st.subheader("🤖 Review AI Predictions")

    st.write(
        "The AI predicts each category. "
        "You can correct any mistakes in the table."
    )

    edited_df = st.data_editor(
        st.session_state["transactions"],
        column_config={
            "Category": st.column_config.SelectboxColumn(
                "Category",
                options=categories,
                required=True
            ),
            "Amount": st.column_config.NumberColumn(
                "Amount ($)",
                min_value=0,
                format="$%.2f"
            )
        },
        disabled=["Description", "Amount"],
        hide_index=True,
        use_container_width=True,
        key="transaction_editor"
    )

    # Use corrected categories in analysis
    df = edited_df.copy()

    # ---------------------------------
    # 6. FINANCIAL DASHBOARD
    # ---------------------------------

    st.divider()
    st.header("📊 Your Financial Dashboard")

    total = df["Amount"].sum()
    remaining = income - total
    budget_remaining = budget - total

    col1, col2, col3 = st.columns(3)

    col1.metric("Monthly Income", f"${income:,.2f}")
    col2.metric("Total Expenses", f"${total:,.2f}")
    col3.metric("Money Remaining", f"${remaining:,.2f}")

    st.subheader("Monthly Budget")

    if budget > 0:
        budget_ratio = total / budget

        st.progress(min(budget_ratio, 1.0))

        st.write(
            f"You have used {budget_ratio:.1%} "
            f"of your monthly spending budget."
        )

        if total > budget:
            st.error(
                f"You are over budget by "
                f"${abs(budget_remaining):,.2f}."
            )
        else:
            st.success(
                f"You have ${budget_remaining:,.2f} "
                f"left in your spending budget."
            )
    else:
        st.info("Set a monthly budget to track spending.")

    # ---------------------------------
    # 7. SPENDING ANALYSIS
    # ---------------------------------

    st.divider()
    st.header("📈 Spending Analysis")

    spending = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spending by Category")
        st.bar_chart(spending)

    with col2:
        st.subheader("Spending Breakdown")
        st.dataframe(
            spending.rename("Total ($)"),
            use_container_width=True
        )

    # ---------------------------------
    # 8. SAVINGS RECOMMENDATIONS
    # ---------------------------------

    st.divider()
    st.header("💡 Smart Budgeting Insights")

    if total > 0:

        largest_category = spending.idxmax()
        largest_amount = spending.max()

        st.write(
            f"Your largest spending category is "
            f"**{largest_category}**, accounting for "
            f"${largest_amount:,.2f}."
        )

        percentage = largest_amount / total * 100

        st.write(
            f"This represents {percentage:.1f}% "
            f"of your total spending."
        )

        reduction = st.slider(
            "Explore a spending reduction (%)",
            min_value=5,
            max_value=30,
            value=10,
            step=5
        )

        potential_savings = (
            largest_amount * reduction / 100
        )

        st.info(
            f"Reducing spending in {largest_category} "
            f"by {reduction}% could save "
            f"${potential_savings:,.2f}."
        )

    else:
        st.info("Add expenses to see budgeting insights.")

    # ---------------------------------
    # 9. SAVINGS GOAL TRACKER
    # ---------------------------------

    st.divider()
    st.header("🎯 Monthly Savings Goal")

    available_savings = max(0.0, remaining)

    if savings_goal > 0:

        goal_progress = min(
            available_savings / savings_goal,
            1.0
        )

        st.progress(goal_progress)

        st.write(
            f"Potential money available for savings: "
            f"**${available_savings:,.2f}**"
        )

        if available_savings >= savings_goal:
            st.success(
                "You have enough remaining income "
                "to meet your savings goal!"
            )
        else:
            shortfall = savings_goal - available_savings

            st.warning(
                f"You are ${shortfall:,.2f} "
                f"short of your savings goal."
            )

    else:
        st.info("Enter a savings goal in the sidebar.")

    # ---------------------------------
    # 10. DOWNLOAD RESULTS
    # ---------------------------------

    st.divider()
    st.header("📥 Download Your Results")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Categorized Transactions",
        data=csv_data,
        file_name="smartspend_results.csv",
        mime="text/csv"
    )

    st.caption(
        "Budgeting suggestions are educational estimates. "
        "AI predictions may contain errors."
    )

else:
    st.info(
        "Upload your expense CSV to start using SmartSpend AI."
    )
