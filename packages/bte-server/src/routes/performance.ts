import path from "path";
import fs from "fs";
import Debug from "debug";
const debug = Debug("bte:biothings-explorer-trapi:performance");
import * as utils from "../utils/common";
import { Express } from "express";

class Performance {
  setRoutes(app: Express) {
    app
      .route("/performance")
      .get((req, res) => {
        debug("start to retrieve performance log.");
        const file_path = path.resolve(__dirname, "../../../../performance-test/report.html");
        debug(`file path is ${file_path}`);
        try {
          fs.access(file_path, fs.constants.R_OK, _err => {
            debug("performance file exists!");
            res.sendFile(file_path);
          });
          // if (fs.existsSync(file_path)) {
          //     debug("performance file exists!")
          //     res.sendFile(file_path)
          // }
        } catch (err) {
          res.setHeader("Content-Type", "application/json");
          res.status(404);
          res.end(JSON.stringify({ error: (err as Error).toString() }));
        }
      })
      .all(utils.methodNotAllowed);
  }
}

export default new Performance();
