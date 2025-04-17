from locust import HttpUser, task, between


class BasicUser(HttpUser):
    wait_time = between(0.5, 3)

    @task
    def query(self):
        self.client.post(
            "/query",
            json={
                "submitter": "bte-dev-tester-manual",
                "message": {
                    "query_graph": {
                        "nodes": {
                            "n0": {
                                "categories": ["biolink:Gene"],
                                "ids": ["NCBIGene:3778"],
                            },
                            "n1": {"categories": ["biolink:Disease"]},
                        },
                        "edges": {
                            "e01": {
                                "subject": "n0",
                                "object": "n1",
                                "predicates": ["biolink:related_to"],
                            }
                        },
                    }
                },
            },
        )
