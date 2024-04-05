import { Telemetry } from "../../src/telemetry";
import * as Sentry from "@sentry/node";

describe("Test telemetry module avoids errors", () => {
  test("Span creation", () => {
    expect(() => {
      const span = Telemetry.startSpan({ description: "fake span" });
      expect(span).toHaveProperty("span");
      span.setData("test", "fakeData");
      span.finish();
    }).not.toThrow();
  });
  test("captureException", () => {
    expect(() => {
      Telemetry.captureException(new Error());
    }).not.toThrow();
  });
  test("addBreadCrumb", () => {
    expect(() => {
      Telemetry.addBreadcrumb({ message: "fake breadcrumb" });
    }).not.toThrow();
  });
});
