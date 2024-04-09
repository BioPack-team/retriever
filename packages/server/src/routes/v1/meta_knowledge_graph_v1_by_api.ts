import handler from "../../controllers/meta_knowledge_graph";
import * as utils from "../../utils/common";
import { Express, NextFunction, Request, Response, RequestHandler } from "express";

class MetaKGByAPI {
  setRoutes(app: Express) {
    app
      .route("/v1/smartapi/:smartapiID/meta_knowledge_graph")
      .get((async (req: Request, res: Response, next: NextFunction) => {
        try {
          const metaKGHandler = new handler(req.params.smartapiID);
          const kg = await metaKGHandler.getKG();
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(kg));
        } catch (error) {
          next(error);
        }
      })as RequestHandler)
      .all(utils.methodNotAllowed);
  }
}

export default new MetaKGByAPI();
