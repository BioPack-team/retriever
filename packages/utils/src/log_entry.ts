import _ from "lodash";
import { TrapiLog } from "./types";
import { Telemetry } from "./telemetry";

export interface LogData {
  type?: string;
  qEdgeID?: string;
  error?: string | number;
  hits?: number;
  [additionalInformation: string]: unknown;
}

export interface StampedLog extends TrapiLog {
  data: LogData;
  toJSON(): TrapiLog;
}

export enum SentryLogSeverity {
  ERROR = "error",
  WARNING = "warning",
  INFO = "info",
  DEBUG = "debug",
}

export class LogEntry {
  level: string;
  code: string | null;
  message: string | null;
  data: LogData | null;
  constructor(
    level = "DEBUG",
    code: string | number | null = null,
    message: string = null,
    data: LogData = null,
  ) {
    this.level = level;
    this.message = message;
    this.code = code === null ? null : String(code);
    this.data = data;
  }

  getLog(): StampedLog {
    const log = {
      timestamp: new Date().toISOString(),
      level: this.level,
      message: this.message,
      code: this.code,
    };
    if (global.job) {
      global.job.log(JSON.stringify(log, undefined, 2));
    }
    Telemetry.addBreadcrumb({
      category: "log",
      message: this.message,
      level: SentryLogSeverity[this.level.toLowerCase()],
    });
    return {
      ...log,
      data: this.data,
      toJSON() {
        return _.omit(this, ["data", "toJSON"]) as StampedLog;
      },
    };
  }
}
