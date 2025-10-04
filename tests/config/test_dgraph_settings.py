from retriever.config.general import DgraphSettings


def test_dgraph_settings_defaults_properties():

    s = DgraphSettings()
    assert s.host == "localhost"
    assert s.http_port == 8080
    assert s.grpc_port == 9080
    assert s.use_tls is False
    assert s.query_timeout == 60
    assert s.connect_retries == 3
    assert s.grpc_max_send_message_length == -1
    assert s.grpc_max_receive_message_length == -1
    assert s.http_endpoint == "http://localhost:8080"
    assert s.grpc_endpoint == "localhost:9080"


def test_dgraph_settings_tls_properties():
    from retriever.config.general import DgraphSettings

    s = DgraphSettings(
        host="example.org",
        http_port=18080,
        grpc_port=19080,
        grpc_max_send_message_length=10,
        grpc_max_receive_message_length=20,
        use_tls=True,
    )
    assert s.http_endpoint == "https://example.org:18080"
    assert s.grpc_endpoint == "example.org:19080"
    assert s.grpc_max_send_message_length == 10
    assert s.grpc_max_receive_message_length == 20


def test_dgraph_settings_runtime_mutation_updates_endpoints():
    from retriever.config.general import DgraphSettings

    s = DgraphSettings()
    assert s.http_endpoint == "http://localhost:8080"
    assert s.grpc_endpoint == "localhost:9080"

    s.host = "dgraph.local"
    s.http_port = 1234
    s.grpc_port = 5678
    s.grpc_max_send_message_length = 10
    s.grpc_max_receive_message_length = 20
    s.use_tls = True

    assert s.http_endpoint == "https://dgraph.local:1234"
    assert s.grpc_endpoint == "dgraph.local:5678"
    assert s.grpc_max_send_message_length == 10
    assert s.grpc_max_receive_message_length == 20
