import Components from "./component";
import Endpoint from "./endpoint";
import Debug from "debug";
const debug = Debug("bte:smartapi-kg:Parser");
import { SmartAPISpec, ParsedAPIMetadataObject, SmartAPIKGOperationObject, APIClass } from "./types";

export default class API implements APIClass {
  private _smartapiDoc: SmartAPISpec;
  /**
   * constructor to load SmartAPI specification.
   * @param {object} smartapiDoc - SmartAPI Specification in Javascript Object format
   */
  constructor(smartapiDoc: SmartAPISpec) {
    this._smartapiDoc = smartapiDoc;
  }

  get smartapiDoc(): SmartAPISpec {
    return this._smartapiDoc;
  }

  get metadata(): ParsedAPIMetadataObject {
    const metadata = this.fetchAPIMeta();
    metadata.operations = this.fetchAllOpts();
    return metadata;
  }
  /**
   * Fetch the title of API from SmartAPI Specification.
   */
  private fetchAPITitle(): string | undefined {
    if (!("info" in this.smartapiDoc)) {
      return undefined;
    }
    return this.smartapiDoc.info.title;
  }

  private fetchXTranslatorInforesCurie(): string | undefined {
    if (!("info" in this.smartapiDoc)) {
      return undefined;
    }
    if (!("x-translator" in this.smartapiDoc.info)) {
      return undefined;
    }
    return this.smartapiDoc.info["x-translator"]["infores"];
  }

  private fetchXTranslatorComponent(): string | undefined {
    if (!("info" in this.smartapiDoc)) {
      return undefined;
    }
    if (!("x-translator" in this.smartapiDoc.info)) {
      return undefined;
    }
    return this.smartapiDoc.info["x-translator"].component;
  }

  private fetchXTranslatorTeam(): string[] {
    if (!("info" in this.smartapiDoc)) {
      return [];
    }
    if (!("x-translator" in this.smartapiDoc.info)) {
      return [];
    }
    return this.smartapiDoc.info["x-translator"].team;
  }

  /**
   * Fetch the tags associated with the API from SmartAPI Specification.
   */
  private fetchAPITags(): string[] {
    if (!("tags" in this.smartapiDoc)) {
      return undefined;
    }
    return this.smartapiDoc.tags.map(x => x.name);
  }

  /**
   * Fetch the url of the server from SmartAPI Specification.
   */
  private fetchServerUrl(): string | undefined {
    if (!("servers" in this.smartapiDoc)) {
      return undefined;
    }

    const productionLevel = process.env.INSTANCE_ENV ?? "";

    const getLevel = (maturity: string) => {
      switch (productionLevel) {
        case "prod":
          if (maturity == "production") return 0;
          return 10000;
        case "test":
          if (maturity == "testing") return 0;
          if (maturity == "production") return 1;
          return 10000;
        case "ci":
          if (maturity == "staging") return 0;
          if (maturity == "testing") return 1;
          if (maturity == "production") return 2;
          return 10000;
        default:
          if (maturity == "development") return 0;
          if (maturity == "staging") return 1;
          if (maturity == "testing") return 2;
          if (maturity == "production") return 3;
          return 10000;
      }
    };

    const servers = this.smartapiDoc.servers.map(server => ({
      url: server.url,
      level: getLevel(server["x-maturity"] ?? "production"),
      maturity: server["x-maturity"] ?? "production",
      https: server.url.includes("https"),
    }));

    const sorted_servers = servers.sort((a, b) => {
      if (a.level != b.level) return a.level - b.level;
      if (a.https != b.https) return a.https ? -1 : 1;
      return 0;
    });

    if (sorted_servers[0].level != 10000) {
      return sorted_servers[0].url;
    }
    debug(`Server ${sorted_servers[0].url} skipped due to insufficient maturity level ${sorted_servers[0].maturity}`);
    return undefined;
  }

  /**
   * Fetch component from SmartAPI Specification.
   */
  private fetchComponents() {
    if (!("components" in this.smartapiDoc)) {
      return undefined;
    }
    return new Components(this.smartapiDoc.components);
  }

  /**
   * Fetch metadata information from SmartAPI Specification.
   */
  private fetchAPIMeta(): ParsedAPIMetadataObject {
    return {
      title: this.fetchAPITitle(),
      tags: this.fetchAPITags(),
      url: this.fetchServerUrl(),
      "x-translator": {
        component: this.fetchXTranslatorComponent(),
        team: this.fetchXTranslatorTeam(),
        infores: this.fetchXTranslatorInforesCurie(),
      },
      smartapi: {
        id: this.smartapiDoc._id,
        meta: this.smartapiDoc._meta,
      },
      components: this.fetchComponents(),
      paths: this.smartapiDoc.paths instanceof Object ? Object.keys(this.smartapiDoc.paths) : [],
      operations: [],
    };
  }

  /**
   * Fetch all operations from SmartAPI Specification.
   */
  private fetchAllOpts(): SmartAPIKGOperationObject[] {
    let ops = [] as SmartAPIKGOperationObject[];
    const apiMeta = this.fetchAPIMeta();
    if ("paths" in this.smartapiDoc) {
      for (const path of Object.keys(this.smartapiDoc.paths)) {
        const ep = new Endpoint(this.smartapiDoc.paths[path], apiMeta, path);
        ops = [...ops, ...ep.constructEndpointInfo()];
      }
    }
    return ops;
  }
}
