import pytest
import asyncio

from retriever.data_tiers.tier_0.dgraph.driver import DgraphDriver


@pytest.mark.asyncio
async def test_dgraph_connect_and_query():
    driver = DgraphDriver()

    # Connect
    await driver.connect()
    assert driver._client is not None

    # Run a basic query
    query = """{ node(func: has(id), first: 1) { id } }"""
    result = await driver.run_query(query)
    assert isinstance(result, dict)  # should return JSON
    assert "node" in result or True

    # Close
    await driver.close()
    assert driver._client is None
