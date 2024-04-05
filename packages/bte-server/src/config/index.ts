import compression from "compression";
import cors from "cors";
import bodyParser from "body-parser";
import rateLimit from "express-rate-limit";
import helmet from "helmet";
import dotenv from "dotenv";
import * as Sentry from "@sentry/node";
import { Express } from "express";
import Debug from "debug";
const debug = Debug("bte:biothings-explorer-trapi:server-config");

export default class Config {
  app: Express;
  constructor(app: Express) {
    this.app = app;
  }

  setConfig() {
    this.setSentry();
    this.setTrustProxy();
    this.setDotEnv();
    this.setNodeEnv();
    this.setBodyParser();
    this.setCors();
    this.setCompression();
    this.setHttpHeaders();
    this.setLimiter();
    return this.app;
  }

  setDotEnv() {
    dotenv.config();
  }

  setNodeEnv() {
    process.env.NODE_ENV = process.env.NODE_ENV || "development";
  }

  setTrustProxy() {
    this.app.set("trust proxy", ["loopback", "linklocal", "uniquelocal"]);
  }

  setBodyParser() {
    // support application/json type post data
    this.app.use(bodyParser.json({ limit: "50mb" }));
    //support application/x-www-form-urlencoded post data
    this.app.use(bodyParser.urlencoded({ limit: "50mb", extended: true }));
    return this.app;
  }

  setCors() {
    const options = {
      allowedHeaders: ["Origin", "X-Requested-With", "Content-Type", "Accept", "X-Access-Token", "Authorization"],
      credentials: true,
      methods: "GET,HEAD,OPTIONS,PUT,PATCH,POST,DELETE",
      origin: "*",
      preflightContinue: false,
    };
    this.app.use(cors(options));
  }

  setCompression() {
    this.app.use(compression());
  }

  setHttpHeaders() {
    this.app.use(
      helmet({
        contentSecurityPolicy: false,
      }),
    );
  }

  setLimiter() {
    const slowLimiter = rateLimit({
      windowMs: 1 * 60 * 1000, //1min
      max: parseInt(process.env.MAX_QUERIES_PER_MIN || "20"),
    });
    const medLimiter = rateLimit({
      windowMs: 1 * 60 * 1000, //1min
      max: parseInt(process.env.MAX_QUERIES_PER_MIN || "30"),
    });
    const fastLimiter = rateLimit({
      windowMs: 1 * 60 * 1000, //1min
      max: parseInt(process.env.MAX_QUERIES_PER_MIN || "6000"),
    });
    this.app.use("/", fastLimiter);
    this.app.use("/v1/query", slowLimiter);
    this.app.use("/v1/team/:team_name/query", medLimiter);
    this.app.use("/v1/team/:smartapiID/query", medLimiter);
    this.app.use("/v1/meta_knowledge_graph", medLimiter);
    this.app.use("/v1/team/:teamName/meta_knowledge_graph", medLimiter);
    this.app.use("/v1/smartapi/:smartapiID/meta_knowledge_graph", medLimiter);
    this.app.use("/v1/asyncquery", fastLimiter);
    this.app.use("/v1/team/:teamName/asyncquery", fastLimiter);
    this.app.use("/v1/smartapi/:smartapiID/asyncquery", fastLimiter);
    this.app.use("/queues", fastLimiter);
  }

  setSentry() {
    // use SENTRY_DSN environment variable
    try {
      Sentry.init({
        integrations: [
          // enable HTTP calls tracing
          new Sentry.Integrations.Http({ tracing: true }),
          // enable Express.js middleware tracing
          new Sentry.Integrations.Express({ app: this.app }),
          // Automatically instrument Node.js libraries and frameworks
          ...Sentry.autoDiscoverNodePerformanceMonitoringIntegrations(),
        ],

        // Set tracesSampleRate to 1.0 to capture 100%
        // of transactions for performance monitoring.
        // We recommend adjusting this value in production
        tracesSampleRate: process.env.EXPRESS_SAMPLE_RATE ? parseFloat(process.env.EXPRESS_SAMPLE_RATE) : 1.0,
        environment: process.env.INSTANCE_ENV,
      });
      // RequestHandler creates a separate execution context, so that all
      // transactions/spans/breadcrumbs are isolated across requests
      this.app.use(Sentry.Handlers.requestHandler({ user: false }));
      // TracingHandler creates a trace for every incoming request
      this.app.use(Sentry.Handlers.tracingHandler());
    } catch (error) {
      debug("Sentry init error. This does not affect execution.");
      debug(error as string);
    }
  }

  setOpenTel() { }
}
