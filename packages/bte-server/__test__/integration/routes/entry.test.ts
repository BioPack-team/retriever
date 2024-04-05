import app from "../../../src/app";
import request from 'supertest';

describe("Test entry point", () => {
    test("query / should display frontpage", async () => {
        await request(app)
            .get("/")
            .expect(200)
            .then((response) => {
                expect(response.text).toContain("<title>BioThings Explorer</title>");
            })
    })
})
