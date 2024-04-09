import handler from "../../controllers/meta_knowledge_graph";
import * as utils from "../../utils/common";
import { Express, NextFunction, Request, Response, RequestHandler } from "express";

class MetaKG {
  setRoutes(app: Express) {
    app
      .route("/v1/meta_knowledge_graph")
      .get((async (_req: Request, res: Response, next: NextFunction) => {
        try {
          const metaKGHandler = new handler(undefined);
          const kg = await metaKGHandler.getKG();
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(kg));
        } catch (error) {
          next(error);
        }
      }) as RequestHandler)
      .all(utils.methodNotAllowed);
  }
}

export default new MetaKG();
