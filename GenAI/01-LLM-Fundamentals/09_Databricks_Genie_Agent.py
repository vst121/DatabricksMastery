
# 09_Databricks_Genie_Agent.py
# Prepare a business dataset for a Databricks Genie Agent.

from pyspark.sql import functions as F

# ---------------------------------------------------------
# 1. Sample business data
# ---------------------------------------------------------

data = [
    ("2026-01-10", "Germany", "Laptop", "Electronics", 120, 1250.0, 900.0),
    ("2026-01-15", "Germany", "Phone", "Electronics", 200, 800.0, 600.0),
    ("2026-02-12", "France", "Laptop", "Electronics", 90, 1250.0, 900.0),
    ("2026-02-18", "France", "Tablet", "Electronics", 150, 500.0, 350.0),
    ("2026-03-05", "Germany", "Laptop", "Electronics", 75, 1250.0, 900.0),
    ("2026-03-20", "Germany", "Phone", "Electronics", 160, 800.0, 600.0),
    ("2026-04-08", "France", "Laptop", "Electronics", 65, 1250.0, 900.0),
    ("2026-04-19", "France", "Tablet", "Electronics", 110, 500.0, 350.0),
    ("2026-05-11", "Germany", "Laptop", "Electronics", 55, 1250.0, 900.0),
    ("2026-05-22", "Germany", "Phone", "Electronics", 130, 800.0, 600.0),
    ("2026-06-03", "France", "Laptop", "Electronics", 45, 1250.0, 900.0),
    ("2026-06-17", "France", "Tablet", "Electronics", 95, 500.0, 350.0),
]

columns = [
    "sale_date",
    "country",
    "product",
    "category",
    "quantity",
    "unit_price",
    "unit_cost",
]

sales_df = spark.createDataFrame(data, columns)

# ---------------------------------------------------------
# 2. Add business metrics
# ---------------------------------------------------------

sales_df = (
    sales_df
    .withColumn("sale_date", F.to_date("sale_date"))
    .withColumn(
        "revenue",
        F.col("quantity") * F.col("unit_price")
    )
    .withColumn(
        "cost",
        F.col("quantity") * F.col("unit_cost")
    )
    .withColumn(
        "profit",
        F.col("revenue") - F.col("cost")
    )
    .withColumn(
        "profit_margin",
        F.col("profit") / F.col("revenue")
    )
)

# ---------------------------------------------------------
# 3. Save as a Delta table
# ---------------------------------------------------------

TABLE_NAME = "genai_sales"

(
    sales_df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(TABLE_NAME)
)

# ---------------------------------------------------------
# 4. Validate the dataset
# ---------------------------------------------------------

print("Created table:", TABLE_NAME)

display(
    spark.table(TABLE_NAME)
    .orderBy("sale_date")
)

# ---------------------------------------------------------
# 5. Basic business validation
# ---------------------------------------------------------

display(
    spark.table(TABLE_NAME)
    .groupBy("country", "product")
    .agg(
        F.sum("quantity").alias("total_units"),
        F.sum("revenue").alias("total_revenue"),
        F.sum("profit").alias("total_profit"),
    )
    .orderBy("country", "product")
)
