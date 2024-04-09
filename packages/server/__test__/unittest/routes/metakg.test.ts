import app from "../../../src/app";
import request from "supertest";
import assoc from "../../../src/controllers/association";
jest.mock("../../../src/controllers/association");
const mockedAssoc = assoc as jest.MockedFunction<typeof assoc>

describe("Test /performance endpoint", () => {
  test("Should return 404 when loading metakg failed", async () => {
    mockedAssoc.mockImplementation(() => {
      throw new Error("Error");
    });
    await request(app)
      .get("/metakg")
      .expect(404)
      .expect("Content-Type", /json/)
      .then(res => {
        expect(res.body).toHaveProperty("description", "Unable to load metakg: Failed to load metakg");
      });
  });
});
