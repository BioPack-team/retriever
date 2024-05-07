import _ from "lodash";
import { TrapiLog } from "./types";
import { Telemetry } from "./telemetry";

export interface LogData {
  type?: string;
  qEdgeID?: string;
  error?: string | number | boolean;
  hits?: number;
  [additionalInformation: string]: unknown;
}

export interface StampedLog extends TrapiLog {
  data: LogData;
  toJSON(): TrapiLog;
}

export interface SerializableLog extends TrapiLog {
  data: LogData;
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

  getSerializeable(isNewLog = true): SerializableLog {
    const log = this.getLog(isNewLog);
    delete log.toJSON;
    return log as SerializableLog;
  }

  static deserialize(logs: SerializableLog[]): StampedLog[] {
    return logs.map(log => {
      return {
        ...log,
        toJSON: () => _.omit(log, ["data", "toJSON"]) as StampedLog,
      } as StampedLog;
    });
  }

  getLog(isNewLog = true): StampedLog {
    const log = {
      timestamp: new Date().toISOString(),
      level: this.level,
      message: this.message,
      code: this.code,
    };
    if (isNewLog) {
      if (global.job) {
        global.job.log(JSON.stringify(log, undefined, 2));
      }

      Telemetry.addBreadcrumb({
        category: "log",
        message: this.message,
        level: SentryLogSeverity[this.level.toLowerCase()],
      });
    }
    return {
      ...log,
      data: this.data,
      toJSON() {
        return _.omit(this, ["data", "toJSON"]) as StampedLog;
      },
    };
  }
}
