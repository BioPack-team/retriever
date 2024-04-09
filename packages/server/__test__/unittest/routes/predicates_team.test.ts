import request from "supertest";
import PredicatesLoadingError from "../../../src/utils/errors/predicates_error";

describe("Test /v1/meta_knowledge_graph endpoint", () => {
  test("Should return 404 with valid response", async () => {
    jest.mock("../../../src/controllers/meta_knowledge_graph.ts", () => ({
      __esModule: true,
      default: jest.fn().mockImplementation(() => {
        throw new PredicatesLoadingError();
      }),
    }));
    const { default: app } = await import("../../../src/app");
    await request(app)
      .get("/v1/team/Text Mining Provider/meta_knowledge_graph")
      .expect(404)
      .expect("Content-Type", /json/)
      .then(res => {
        expect(res.body).toHaveProperty("description", "Unable to load predicates: Failed to load metakg");
      });
  });
});
