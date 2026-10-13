import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="AI Data Risk Platform",
    page_icon="📊",
    layout="wide"
)

DB_NAME = "risk_results.db"


def init_database():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analyzed_at TEXT NOT NULL,
                total_transactions INTEGER NOT NULL,
                unusual_transactions INTEGER NOT NULL,
                average_risk_score REAL
            )
        """)


def save_result(total, unusual, average_risk):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            INSERT INTO risk_results
            (analyzed_at, total_transactions,
             unusual_transactions, average_risk_score)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            int(total),
            int(unusual),
            float(average_risk) if pd.notna(average_risk) else None
        ))


def load_results():
    with sqlite3.connect(DB_NAME) as conn:
        return pd.read_sql_query(
            "SELECT * FROM risk_results ORDER BY id DESC",
            conn
        )


init_database()

st.title("AI Data Risk Platform")
st.subheader("Intelligent Data Analysis and Risk Detection")
st.write("Upload your data, analyze it, and discover unusual patterns.")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.size == 0:
            st.error("The CSV file is empty. Please upload a file containing data.")
        else:
            df = pd.read_csv(uploaded_file)

            if df.empty or len(df.columns) == 0:
                st.error("The CSV file contains no data. Please upload a valid file.")
            else:
                st.subheader("Original Data")
                st.dataframe(df, use_container_width=True)

                original_rows = len(df)
                df = df.drop_duplicates()
                df = df.dropna(how="all").reset_index(drop=True)

                st.subheader("Data Cleaning")
                st.write(f"Rows before cleaning: {original_rows}")
                st.write(f"Rows after cleaning: {len(df)}")
                st.write(f"Duplicate rows removed: {original_rows - len(df)}")
                st.dataframe(df, use_container_width=True)

                st.subheader("Data Analysis")

                if "transaction_amount" in df.columns:
                    amounts = pd.to_numeric(
                        df["transaction_amount"], errors="coerce"
                    )
                    total_amount = amounts.sum()
                else:
                    amounts = pd.Series(dtype=float)
                    total_amount = 0

                if "risk_score" in df.columns:
                    risks = pd.to_numeric(df["risk_score"], errors="coerce")
                    average_risk = risks.mean()
                else:
                    risks = pd.Series(dtype=float)
                    average_risk = float("nan")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Transactions", len(df))
                c2.metric("Total Amount", f"{total_amount:,.0f}")
                c3.metric(
                    "Average Risk Score",
                    f"{average_risk:.1f}"
                    if pd.notna(average_risk) else "N/A"
                )

                if "date" in df.columns and "transaction_amount" in df.columns:
                    chart_df = df[["date", "transaction_amount"]].copy()
                    chart_df["date"] = pd.to_datetime(
                        chart_df["date"], errors="coerce"
                    )
                    chart_df["transaction_amount"] = pd.to_numeric(
                        chart_df["transaction_amount"], errors="coerce"
                    )
                    chart_df = chart_df.dropna().sort_values("date")

                    if not chart_df.empty:
                        st.subheader("Transaction Amounts")
                        st.bar_chart(
                            chart_df.set_index("date")["transaction_amount"]
                        )

                if "date" in df.columns and "risk_score" in df.columns:
                    risk_chart = df[["date", "risk_score"]].copy()
                    risk_chart["date"] = pd.to_datetime(
                        risk_chart["date"], errors="coerce"
                    )
                    risk_chart["risk_score"] = pd.to_numeric(
                        risk_chart["risk_score"], errors="coerce"
                    )
                    risk_chart = risk_chart.dropna().sort_values("date")

                    if not risk_chart.empty:
                        st.subheader("Risk Scores")
                        st.line_chart(
                            risk_chart.set_index("date")["risk_score"]
                        )

                if "transaction_type" in df.columns:
                    st.subheader("Transactions by Type")
                    st.bar_chart(df["transaction_type"].value_counts())

                st.subheader("AI Risk Detection")

                required = ["transaction_amount", "risk_score"]

                if all(column in df.columns for column in required):
                    ai_data = df[required].copy()

                    for column in required:
                        ai_data[column] = pd.to_numeric(
                            ai_data[column], errors="coerce"
                        )

                    valid = ai_data.dropna()

                    if len(valid) >= 3:
                        model = IsolationForest(
                            contamination=0.2,
                            random_state=42
                        )
                        predictions = model.fit_predict(valid)

                        ai_results = df.loc[valid.index].copy()
                        ai_results["AI Result"] = [
                            "Unusual" if p == -1 else "Normal"
                            for p in predictions
                        ]

                        unusual_count = int(
                            (ai_results["AI Result"] == "Unusual").sum()
                        )

                        st.dataframe(ai_results, use_container_width=True)
                        st.metric(
                            "Unusual Transactions Detected", unusual_count
                        )
                        st.warning(
                            "AI results are indicators for review, not proof of fraud."
                        )

                        if st.button("Save Analysis to Database"):
                            save_result(
                                len(df), unusual_count, average_risk
                            )
                            st.success("Analysis saved to the database!")
                    else:
                        st.warning(
                            "At least 3 valid transactions are required for AI detection."
                        )
                else:
                    st.warning(
                        "The CSV must contain transaction_amount and risk_score."
                    )

    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        st.error(
            "Could not read this CSV file. Please upload a valid CSV containing data."
        )
    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.divider()
st.subheader("Saved Analysis History")

history = load_results()

if history.empty:
    st.info("No saved analysis results yet.")
else:
    st.dataframe(history, use_container_width=True)