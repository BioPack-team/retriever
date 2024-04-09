import swaggerValidation from "./validate";
import { InvalidQueryGraphError } from "@biothings-explorer/query_graph_handler";
import PredicatesLoadingError from "../utils/errors/predicates_error";
import MetaKGLoadingError from "../utils/errors/metakg_error";
import ServerOverloadedError from "../utils/errors/server_overloaded_error";
import Debug from "debug";
const debug = Debug("bte:biothings-explorer-trapi:error_handler");
import * as Sentry from "@sentry/node";
import { Express, NextFunction, Request, Response } from "express";
import StatusError from "../utils/errors/status_error";
import { TrapiResponse } from "@biothings-explorer/types";

class ErrorHandler {
  shouldHandleError(error: Error) {
    // Capture all 404 and 500 errors
    if (error instanceof swaggerValidation.InputValidationError || error.name === "InputValidationError") {
      return false;
    }
    if (
      error instanceof InvalidQueryGraphError ||
      error.stack.includes("InvalidQueryGraphError") ||
      error.name === "InvalidQueryGraphError"
    ) {
      return false;
    }
    if (error.name === "QueryAborted") {
      return false;
    }
    return true;
  }

  setRoutes(app: Express) {
    // first pass through sentry
    try {
      app.use(
        Sentry.Handlers.errorHandler({
          shouldHandleError(error) {
            // Do not capture non-server errors
            if (error.status && Number(error.status) < 500) {
              return false;
            }
            if (error instanceof swaggerValidation.InputValidationError || error.name === "InputValidationError") {
              return false;
            }
            if (
              error instanceof InvalidQueryGraphError ||
              error.stack.includes("InvalidQueryGraphError") ||
              error.name === "InvalidQueryGraphError"
            ) {
              return false;
            }
            if (error.name === "QueryAborted") {
              return false;
            }
            return true;
          },
        }),
      );
    } catch (error) {
      debug("Sentry express config error. This does not affect execution.");
      debug(error);
    }

    app.use((error: Error, req: Request, res: Response, _next: NextFunction) => {
      const json = {
        status: "QueryNotTraversable",
        description: error.message,
      };
      if (error instanceof swaggerValidation.InputValidationError || error.name === "InputValidationError") {
        json.description = `Your input query graph is invalid. Errors: ${(
          error as swaggerValidation.InputValidationError
        ).errors.join("\n")}`;
        return res.status(400).json(json);
      }
      // read stack when instance or err is broken
      if (
        error instanceof InvalidQueryGraphError ||
        error.stack.includes("InvalidQueryGraphError") ||
        error.name === "InvalidQueryGraphError"
      ) {
        json.description = `Your input query graph is invalid: ${error.message}`;
        return res.status(400).json(json);
      }
      if (error instanceof PredicatesLoadingError || error.name === "PredicatesLoadingError") {
        json.status = "KPsNotAvailable";
        json.description = `Unable to load predicates: ${error.message}`;
        return res.status(404).json(json);
      }

      if (error instanceof MetaKGLoadingError || error.name === "MetaKGLoadingError") {
        json.status = "KPsNotAvailable";
        json.description = `Unable to load metakg: ${error.message}`;
        return res.status(404).json(json);
      }

      if (error instanceof ServerOverloadedError || error.name === "ServerOverloadedError") {
        return res
          .status(503)
          .set("Retry-After", String((error as ServerOverloadedError).retryAfter))
          .json(json);
      }
      if (!(error as StatusError).statusCode) (error as StatusError).statusCode = 500;

      if ((error as StatusError).statusCode === 301) {
        return res.status(301).redirect("/");
      }
      debug(error);
      if (req.originalUrl.includes("asyncquery")) {
        return res.status((error as StatusError).statusCode).json({
          status: (error as StatusError).statusCode,
          description: error.toString(),
          trace: process.env.NODE_ENV === "production" ? undefined : error.stack,
        });
      }
      return res.status((error as StatusError).statusCode).json({
        message: {
          query_graph: (req.body as TrapiResponse)?.message?.query_graph,
          knowledge_graph: { nodes: {}, edges: {} },
          results: [],
        },
        status: (error as StatusError).statusCode,
        description: error.toString(),
        trace: process.env.NODE_ENV === "production" ? undefined : error.stack,
      });
    });
  }
}

export default new ErrorHandler();
