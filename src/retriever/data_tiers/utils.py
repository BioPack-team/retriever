from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.base_transpiler import Transpiler
from retriever.data_tiers.tier_0.base_query import Tier0Query
from retriever.data_tiers.tier_0.dgraph.driver import DgraphGrpcDriver
from retriever.data_tiers.tier_0.dgraph.query import DgraphQuery
from retriever.data_tiers.tier_0.dgraph.transpiler import DgraphTranspiler
from retriever.data_tiers.tier_0.neo4j.driver import Neo4jDriver
from retriever.data_tiers.tier_0.neo4j.query import Neo4jQuery
from retriever.data_tiers.tier_0.neo4j.transpiler import Neo4jTranspiler
from retriever.data_tiers.tier_1.elasticsearch.driver import ElasticSearchDriver
from retriever.data_tiers.tier_1.elasticsearch.transpiler import ElasticsearchTranspiler

BACKEND_DRIVERS = dict[str, DatabaseDriver](
    elasticsearch=ElasticSearchDriver(),
    neo4j=Neo4jDriver(),
    dgraph=DgraphGrpcDriver(),
)

TRANSPILERS = dict[str, Transpiler](
    elasticsearch=ElasticsearchTranspiler(),
    neo4j=Neo4jTranspiler(),
    dgraph=DgraphTranspiler(),
)

QUERY_HANDLERS = dict[str, type[Tier0Query]](
    neo4j=Neo4jQuery,
    dgraph=DgraphQuery,
)
