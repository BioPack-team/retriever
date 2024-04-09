import handler from "../../controllers/meta_knowledge_graph";
import * as utils from "../../utils/common";
import { Express, NextFunction, Request, Response, RequestHandler } from "express";

class MetaKGByTeam {
  setRoutes(app: Express) {
    app
      .route("/v1/team/:teamName/meta_knowledge_graph")
      .get((async (req: Request, res: Response, next: NextFunction) => {
        try {
          const metaKGHandler = new handler(undefined, req.params.teamName);
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

export default new MetaKGByTeam();
