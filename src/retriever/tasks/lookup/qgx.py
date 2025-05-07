from reasoner_pydantic import (
    AsyncQuery,
    LogEntry,
    Query,
    Results,
)


async def execute_query_graph(
    query: Query | AsyncQuery,
) -> tuple[Results, list[LogEntry]]:
    # TODO implement actual query handling
    return Results(), []
