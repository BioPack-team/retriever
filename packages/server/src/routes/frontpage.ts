import history from "connect-history-api-fallback";
import express from "express";
import path from "path";
import { Express } from "express";

class FrontPage {
  setRoutes(app: Express) {
    const staticFileMiddleware = express.static(path.resolve(__dirname, "../../../web-app/dist"));
    app.use(staticFileMiddleware);
    app.use(
      history({
        disableDotRule: true,
      }),
    );
    app.use(staticFileMiddleware);
  }
}

export default new FrontPage();
