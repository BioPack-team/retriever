import { TrapiLog } from "@retriever/types";

export default class StatusError extends Error {
  statusCode: number;
  logs?: TrapiLog[];
  constructor(message: string, ...params: string[]) {
    super(...params);

    this.message = message;
  }
}
