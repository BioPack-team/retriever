import { TrapiLog } from "@biothings-explorer/types";

export default class StatusError extends Error {
  statusCode: number;
  logs?: TrapiLog[];
  constructor(message: string, ...params: string[]) {
    super(...params);

    this.message = message;
  }
}
