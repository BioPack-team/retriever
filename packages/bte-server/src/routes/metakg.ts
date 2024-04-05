import * as utils from "../utils/common";
import assoc from "../controllers/association";
import MetaKGLoadingError from "../utils/errors/metakg_error";
import { Express, NextFunction, Request, Response, RequestHandler } from "express";

class MetaKG {
  setRoutes(app: Express) {
    app
      .route("/metakg")
      .get((async (req: Request, res: Response, next: NextFunction) => {
        try {
          res.setHeader("Content-Type", "application/json");
          let api: string = undefined,
            source: string = undefined;
          if (req.query.api !== undefined) {
            api = utils.removeQuotesFromQuery(req.query.api as string);
          }
          if (req.query.provided_by !== undefined) {
            source = utils.removeQuotesFromQuery(req.query.provided_by as string);
          }
          const assocs = assoc(
            req.query.subject as string,
            req.query.object as string,
            req.query.predicate as string,
            req.query.component as string,
            api,
            source,
          );
          res.end(JSON.stringify({ associations: assocs }));
        } catch (error) {
          next(new MetaKGLoadingError());
        }
      }) as RequestHandler)
      .all(utils.methodNotAllowed);
  }
}

export default new MetaKG();
