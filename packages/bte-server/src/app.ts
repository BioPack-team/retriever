import express, { Express } from "express";
import Config from "./config/index";
import { setRoutes } from "./routes/index";

class App {
  app: Express;
  config: Config;
  constructor() {
    this.app = express();
    this.config = new Config(this.app);
    this.app = this.config.setConfig();
    setRoutes(this.app);
  }
}

export default new App().app;
