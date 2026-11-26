print("Python execution is working.")

print("Spark version:", spark.version)

print(
    "Current user:",
    spark.sql("SELECT current_user()").first()[0]
)

from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

print("Workspace:", workspace.config.host)

headers = workspace.config.authenticate()

print("Authentication successful:", "Authorization" in headers)