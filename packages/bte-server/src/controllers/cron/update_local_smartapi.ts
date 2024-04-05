import Debug from "debug";
const debug = Debug("bte:biothings-explorer-trapi:cron");
import axios from "axios";
import fs from "fs/promises";
import path from "path";
import cron from "node-cron";
import { readFile } from "fs/promises";
import yaml from "js-yaml";
import url from "url";
import validUrl from "valid-url";
import SMARTAPI_EXCLUSIONS from "../../config/smartapi_exclusions";
import getSmartApiOverrideConfig from "../../config/smartapi_overrides";
import { SmartApiOverrides } from "../../types";

const userAgent = `BTE/${process.env.NODE_ENV === "production" ? "prod" : "dev"} Node/${process.version} ${
  process.platform
}`;

interface Server {
  url: string;
}

interface Info {
  title: string;
  "x-translator": {
    component: string;
    team: string;
    infores: string;
  };
  "x-trapi": {
    version: string;
    batch_size_limit: number;
    rate_limit: number;
  };
}

interface Paths {
  [path: string]: unknown;
}

interface Tag {
  name: string;
}

interface Spec {
  _id: string;
  _meta: unknown;
  _score: number;
  tags: Tag[];
  info: Info;
  servers: Server[];
  paths: Paths;
}

interface API {
  association: {
    api_name: string;
    smartapi: {
      id: string;
      meta: unknown;
    };
    "x-translator": {
      component: string;
      team: string;
      infores: string;
    };
    "x-trapi": {
      batch_size_limit: number;
      rate_limit: number;
    };
  };
  tags: string[];
  query_operation: {
    path: string;
    server: string;
    method: string;
  };
}

function getServerFromSpec(spec: Spec): string {
  const productionLevel = process.env.INSTANCE_ENV ?? "";

  const getLevel = (maturity: string) => {
    switch (productionLevel) {
      case "test":
        if (maturity == "testing") return 0;
        if (maturity == "production") return 1;
        return 10000;
      case "ci":
        if (maturity == "staging") return 0;
        if (maturity == "testing") return 1;
        if (maturity == "production") return 2;
        return 10000;
      case "dev":
        if (maturity == "development") return 0;
        if (maturity == "staging") return 1;
        if (maturity == "testing") return 2;
        if (maturity == "production") return 3;
        return 10000;
      default:
        if (maturity == "production") return 0;
        return 10000;
    }
  };

  const servers = spec.servers.map(server => ({
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

  if (sorted_servers[0].level == 10000) {
    throw new Error(
      `Server ${sorted_servers[0].url} skipped due to insufficient maturity level ${sorted_servers[0].maturity}`,
    );
  }
  return sorted_servers[0].url;
}

function sortObject(object: unknown) {
    if (Array.isArray(object)) {
        return object.sort();
    }

    // apparently typeof null is object
    if (typeof object === 'object' && object !== null) {
        return Object.keys(object).sort().reduce((acc, key) => {
            acc[key] = sortObject(object[key]);
            return acc;
        }, {});
    }

    return object;
}
function getTRAPIWithPredicatesEndpoint(specs: Spec[]): API[] {
  const trapi: API[] = [];
  let excluded_list = SMARTAPI_EXCLUSIONS.map(api => api.id);
  specs.map(spec => {
    try {
      if (
        "info" in spec &&
        "x-translator" in spec.info &&
        spec.info["x-translator"].component === "KP" &&
        "paths" in spec &&
        "/query" in spec.paths &&
        "x-trapi" in spec.info &&
        spec.servers.length &&
        "/meta_knowledge_graph" in spec.paths &&
        !excluded_list.includes(spec._id) &&
        getServerFromSpec(spec)
      ) {
        let api: API = {
          association: {
            api_name: spec.info.title,
            smartapi: {
              id: spec._id,
              meta: spec._meta,
            },
            "x-translator": {
              component: "KP",
              team: spec.info["x-translator"].team,
              infores: spec.info["x-translator"].infores,
            },
            "x-trapi": {
              batch_size_limit: spec.info["x-trapi"].batch_size_limit,
              rate_limit: spec.info["x-trapi"].rate_limit,
            },
          },
          tags: spec.tags.map(item => item.name),
          query_operation: {
            path: "/query",
            server: getServerFromSpec(spec),
            method: "post",
          },
        };
        // check TRAPI latest accepted version
        if ("/meta_knowledge_graph" in spec.paths) {
          if (
            Object.prototype.hasOwnProperty.call(spec.info["x-trapi"], "version") &&
            spec.info["x-trapi"].version.includes("1.4")
          ) {
            api["predicates_path"] = "/meta_knowledge_graph";
            trapi.push(api);
          }
        } else {
          debug(`[error]: Unable to parse spec, ${spec ? spec.info.title : spec}. Endpoint required not found.`);
        }
      }
    } catch (err) {
      debug(`[error]: Unable to parse spec, ${spec ? spec.info.title : spec}. Error message is ${err.toString()}`);
    }
  });
  return trapi;
}

function constructQueryUrl(serverUrl: string, path: string) {
  if (serverUrl.endsWith("/")) {
    serverUrl = serverUrl.slice(0, -1);
  }
  return serverUrl + path;
}

function getPredicatesFromGraphData(predicate_endpoint: string, data) {
  //if /predicates just return normal response
  if (predicate_endpoint !== "/meta_knowledge_graph") {
    return data;
  }
  // transform graph data to legacy format > object.subject : predicates
  const predicates = {};

  const addNewPredicates = edge => {
    if (edge.knowledge_types && Array.isArray(edge.knowledge_types)) {
      if (!edge.knowledge_types.includes("lookup")) {
        return;
      }
    }
    if (!Object.prototype.hasOwnProperty.call(predicates, edge.object)) {
      predicates[edge.object] = {};
    }
    if (Array.isArray(predicates[edge.object][edge.subject])) {
      predicates[edge.object][edge.subject].push({ predicate: edge.predicate, qualifiers: edge.qualifiers });
    } else {
      predicates[edge.object][edge.subject] = [{ predicate: edge.predicate, qualifiers: edge.qualifiers }];
    }
  };

  if (Object.prototype.hasOwnProperty.call(data, "edges")) {
    data.edges.forEach(edge => addNewPredicates(edge));
  } else {
    //some apis still redirect to previous format
    return data;
  }
  return predicates;
}

async function getOpsFromEndpoint(metadata) {
  return axios
    .get(constructQueryUrl(metadata.query_operation.server, metadata.predicates_path), { timeout: 10000 })
    .then(res => {
      if (res.status === 200) {
        debug(`Successfully got ${metadata.predicates_path} for ${metadata.query_operation.server}`);
        return {
          ...metadata,
          ...{ predicates: getPredicatesFromGraphData(metadata.predicates_path, res.data) },
          nodes: res.data.nodes,
        };
      }
      debug(
        `[error]: API "${metadata.association.api_name}" Unable to get ${metadata.predicates_path}` +
          ` for ${metadata.query_operation.server} due to query failure with status code ${res.status}`,
      );
      return false;
    })
    .catch(err => {
      debug(
        `[error]: API "${metadata.association.api_name}" failed to get ${metadata.predicates_path} for ${
          metadata.query_operation.server
        } due to error ${err.toString()}`,
      );
      return false;
    });
}

async function getOpsFromPredicatesEndpoints(specs: Spec[]) {
  const metadatas = getTRAPIWithPredicatesEndpoint(specs);
  metadatas.sort((a, b) => a.association.smartapi.id.localeCompare(b.association.smartapi.id));
  let res = [];
  debug(`Now caching predicates from ${metadatas.length} TRAPI APIs`);
  await Promise.allSettled(metadatas.map(metadata => getOpsFromEndpoint(metadata))).then(results => {
    results.map(rec => {
      if (rec.status === "fulfilled" && rec.value) {
        res.push(sortObject(rec.value));
      }
    });
  });
  debug(`Got ${res.length} successful requests`);
  return res;
}

async function updateSmartAPISpecs() {
  const SMARTAPI_URL =
    "https://smart-api.info/api/query?q=tags.name:translator&size=1000&sort=_seq_no&raw=1&fields=paths,servers,tags,components.x-bte*,info,_meta";
  let overrides: SmartApiOverrides;
  try {
    overrides = getSmartApiOverrideConfig();
  } catch (error) {
    debug(`ERROR getting API Overrides file because ${error}`);
    return;
  }

  let res: {
    data: {
      total: number;
      hits: Spec[];
    };
  } = { data: { total: 0, hits: [] } };

  if (!(process.env.API_OVERRIDE === "true" && overrides.config.only_overrides)) {
    res = await axios.get(SMARTAPI_URL, { headers: { "User-Agent": userAgent } }).catch(err => {
      debug(`SmartAPI request failed.`);
      throw err;
    });
  }

  const localFilePath = path.resolve(__dirname, "../../../data/smartapi_specs.json");
  const predicatesFilePath = path.resolve(__dirname, "../../../data/predicates.json");
  if (process.env.API_OVERRIDE === "true") {
    await getAPIOverrides(res.data, overrides);
  }

  debug(`Retrieved ${res.data.total} SmartAPI records`);
  //clean _score fields
  const hits = res.data.hits;
  hits.forEach((obj: { _score: any }) => {
    delete obj._score;
  });

  await fs.writeFile(localFilePath, JSON.stringify({ hits: hits }));
  const predicatesInfo = await getOpsFromPredicatesEndpoints(res.data.hits);
  await fs.writeFile(predicatesFilePath, JSON.stringify(predicatesInfo));
}

async function getAPIOverrides(data: { total?: number; hits: any }, overrides: SmartApiOverrides) {
  // if only_overrides is enabled, only overridden apis are used
  if (overrides.config.only_overrides) {
    debug("Override specifies removal of undeclared APIs");
    data.hits = [];
  }
  await Promise.all(
    Object.keys(overrides.apis).map(async id => {
      let override: Spec;
      try {
        const filepath = path.resolve(url.fileURLToPath(overrides.apis[id]));
        override = yaml.load(await readFile(filepath, { encoding: "utf8" })) as Spec;
      } catch (error) {
        if (error instanceof TypeError) {
          if (validUrl.isWebUri(overrides.apis[id])) {
            try {
              override = yaml.load(
                (
                  await axios.get(overrides.apis[id], {
                    headers: { "User-Agent": userAgent },
                  })
                ).data,
              ) as Spec;
            } catch (weberror) {
              debug(`ERROR getting URL-hosted override for API ID ${id} because ${weberror}`);
              return;
            }
          } else {
            try {
              const filepath = path.resolve(overrides.apis[id]);
              override = yaml.load(await readFile(filepath, { encoding: "utf8" })) as Spec;
            } catch (filerror) {
              debug(`ERROR getting local file override for API ID ${id} because ${filerror}`);
              return;
            }
          }
        } else {
          debug(`ERROR getting 'file:///' override for API ID ${id} because ${error}`);
          return;
        }
      }

      debug(`Successfully got override ${id} from ${overrides.apis[id]}`);
      override._id = id;
      override._meta = {
        date_created: undefined,
        last_updated: undefined,
        url: overrides.apis[id],
        username: undefined,
      };
      const index = overrides.config.only_overrides ? -2 : data.hits.findIndex((hit: Spec) => hit._id === id);
      if (index < 0) {
        if (index === -1) debug(`[warning] Overridden API ID ${id} not recognized, appending as new API hit.`);
        data.hits.push(override);
      } else {
        data.hits[index] = override;
      }
      return;
    }),
  );
}

export default function manageSmartApi() {
  // Env set in manual sync script
  const sync_and_exit = process.env.SYNC_AND_EXIT === "true";
  if (sync_and_exit) {
    debug("Syncing SmartAPI specs with subsequent exit...");
    updateSmartAPISpecs()
      .then(() => {
        debug("SmartAPI sync successful.");
        process.exit(0);
      })
      .catch(err => {
        debug(`Updating local copy of SmartAPI specs failed! The error message is ${err.toString()}`);
      });
    return;
  }

  const should_sync =
    process.env.SMARTAPI_SYNC === "true" ||
    [
      process.env.SMARTAPI_SYNC !== "false", // Shouldn't be explicitly disabled
      process.env.NODE_ENV === "production", // Should be in production mode
      process.env.INSTANCE_ID !== "0", // Only one PM2 cluster instance should sync
    ].every(condition => condition);

  if (!should_sync) {
    debug(`SmartAPI sync disabled, server process ${process.pid} disabling smartapi updates.`);
    return;
  }

  // Otherwise, schedule sync!
  cron.schedule("*/10 * * * *", async () => {
    debug(`Updating local copy of SmartAPI specs now at ${new Date().toUTCString()}!`);
    try {
      await updateSmartAPISpecs();
      debug("Successfully updated the local copy of SmartAPI specs.");
    } catch (err) {
      debug(`Updating local copy of SmartAPI specs failed! The error message is ${err.toString()}`);
    }
  });

  // Run at start once
  debug(`Running initial update of SmartAPI specs now at ${new Date().toUTCString()}`);
  updateSmartAPISpecs()
    .then(() => {
      debug("SmartAPI sync successful.");
      process.exit(0);
    })
    .catch(err => {
      debug(`Updating local copy of SmartAPI specs failed! The error message is ${err.toString()}`);
    });
}
