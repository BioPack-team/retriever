import { transform } from "../json_transform/index";
import { JSONDoc } from "../json_transform/types";
import { generateCurie, toArray } from "../utils";
import { RetrieverQueryObject } from "../types";
import _ from "lodash";
import Debug from "debug";
import async from "async";
import { Record, APIEdge } from "@retriever/graph";
const debug = Debug("retriever:api-response-transform:transformer");

export default class BaseTransformer {
  protected edge: APIEdge;
  protected data: RetrieverQueryObject;
  public config: any;

  constructor(data: RetrieverQueryObject, config: any) {
    this.data = data;
    this.edge = data.edge;
    this.config = config;
  }

  /**
   * Create an object with key representing input, and value representing the output of API
   */
  async pairCurieWithAPIResponse() {
    const input = generateCurie(
      this.edge.association.input_id as string,
      this.edge.input.hasOwnProperty("queryInputs")
        ? this.edge.input["queryInputs"]
        : (this.edge.input as string),
    );
    return {
      [input]: [this.data.response],
    };
  }

  /**
   * Wrapper functions to transform API response before passing to JSON Transformer
   * @return {Object} - key is curie representing input, value is an array of outputs.
   */
  wrap(res) {
    if (Array.isArray(res)) {
      res = { data: res };
    }
    return res;
  }

  /**
   * Transform Individual JSON response into Biolink compatible format
   * @param {Object} res - JSON response representing an output.
   */
  jsonTransform(res: JSONDoc | JSONDoc[]) {
    res = transform(res, this.edge.response_mapping);
    return res;
  }

  _updatePublications(mappedResponse: any) {
    if (!Array.isArray(mappedResponse.publications)) {
      mappedResponse.publications = [];
    }

    const publicationTypes = [
      {
        prop: "ref_pmid",
        prefix: "PMID:",
        urls: [
          "http://www.ncbi.nlm.nih.gov/pubmed/",
          "http://europepmc.org/abstract/MED/",
          "https://www.ncbi.nlm.nih.gov/pubmed/",
        ],
      },
      {
        prop: "ref_pmcid",
        prefix: "PMCID:",
        urls: [
          "http://www.ncbi.nlm.nih.gov/pmc/articles/",
          "http://europepmc.org/articles/",
        ],
      },
      {
        prop: "ref_clinicaltrials",
        prefix: "clinicaltrials:",
        urls: [
          "https://clinicaltrials.gov/ct2/show/",
          "https://www.clinicaltrials.gov/ct2/show/",
        ],
      },
      {
        prop: "ref_doi",
        prefix: "doi:",
        urls: [
          "https://doi.org/",
          "http://www.nejm.org/doi/full/",
          "https://www.tandfonline.com/doi/abs/",
          "http://onlinelibrary.wiley.com/doi/",
        ],
      },
      {
        prop: "ref_isbn",
        prefix: "isbn:",
        urls: ["https://www.isbn-international.org/identifier/"],
      },
    ];

    // handle URLs (which could be CURIEs)
    if ("ref_url" in mappedResponse) {
      for (const publication of toArray(mappedResponse.ref_url)) {
        if (typeof publication !== "string" || publication.length === 0) {
          continue;
        }

        let isCurie = false;
        for (const publicationType of publicationTypes) {
          for (const url of publicationType.urls) {
            if (publication.startsWith(url)) {
              isCurie = true;

              if (!mappedResponse[publicationType.prop]) {
                mappedResponse[publicationType.prop] = [];
              } else if (!Array.isArray(mappedResponse[publicationType.prop])) {
                mappedResponse[publicationType.prop] = toArray(
                  mappedResponse[publicationType.prop],
                );
              }

              mappedResponse[publicationType.prop].push(
                publication.slice(url.length),
              );

              break;
            }
          }

          if (isCurie) {
            break;
          }
        }

        if (!isCurie) {
          mappedResponse.publications.push(publication);
        }
      }
    }
    delete mappedResponse.ref_url;

    for (const publicationType of publicationTypes) {
      if (publicationType.prop in mappedResponse) {
        for (let publication of toArray(mappedResponse[publicationType.prop])) {
          // handle numbers
          if (typeof publication === "number") {
            publication = publication.toString();
          }

          if (typeof publication !== "string" || publication.length === 0) {
            continue;
          }

          if (
            publication
              .toUpperCase()
              .startsWith(publicationType.prefix.toUpperCase())
          ) {
            mappedResponse.publications.push(
              publicationType.prefix +
              publication.slice(publicationType.prefix.length),
            );
          } else {
            mappedResponse.publications.push(
              publicationType.prefix + publication,
            );
          }
        }

        delete mappedResponse[publicationType.prop];
      }
    }

    if (mappedResponse.publications.length === 0) {
      delete mappedResponse.publications;
    }

    return mappedResponse;
  }

  // _updateEdgeMetadata(res) {
  //     res.$edge_metadata = {
  //         ...this.edge.association,
  //         trapi_qEdge_obj: this.edge.reasoner_edge,
  //         filter: this.edge.filter,
  //     };
  //     return res;
  // }

  _getSubject(subjectCurie: any) {
    //debug(`input: ${input}`);
    let subject = {
      original:
        typeof this.edge.original_input === "undefined"
          ? undefined
          : this.edge.original_input[subjectCurie],
      // normalizedInfo:
      //     typeof this.edge.input_resolved_identifiers === "undefined" || typeof this.edge.original_input === "undefined"
      //     ? undefined
      //     : this.edge.input_resolved_identifiers[this.edge.original_input[subjectCurie]],
    };
    if (
      this.edge.input_resolved_identifiers &&
      subject.original === undefined
    ) {
      //try to find an equivalent ids object if the original input doesn't match (for ICEES)
      for (const curie of Object.keys(this.edge.input_resolved_identifiers)) {
        if (
          this.edge.input_resolved_identifiers[curie].equivalentIDs.includes(
            subjectCurie,
          )
        ) {
          subject = {
            original: curie,
            // normalizedInfo: this.edge.input_resolved_identifiers[curie],
          };
          break;
        }
      }
    }
    return subject;
  }

  _removeNonEdgeData(mappedResponse: any) {
    delete mappedResponse["@type"];
    delete mappedResponse[this.edge.association.output_id as string];
    delete mappedResponse["input_name"];
    delete mappedResponse["output_name"];
    return mappedResponse;
  }

  /**
   * Add edge information into individual output JSON.
   * @param {Object} mappedResponse - JSON response representing an output.
   */
  async formatRecords(subjectCurie: string, mappedResponse: any) {
    if (
      mappedResponse === undefined ||
      Object.keys(mappedResponse).length === 0
    ) {
      return [];
    }

    // mappedResponse = this._updateEdgeMetadata(mappedResponse);
    const objectIDs = this.extractObjectIDs(mappedResponse);
    const outputName = mappedResponse.output_name;
    const inputName = mappedResponse.input_name;
    mappedResponse = this._removeNonEdgeData(mappedResponse);
    mappedResponse = this._updatePublications(mappedResponse);

    const frozenRecord = {
      subject: { ...this._getSubject(subjectCurie), apiLabel: inputName },
      association: { ...this.edge.association },
      qEdge: this.edge.reasoner_edge,
      mappedResponse: { ...mappedResponse },
    };

    const transformedRecords = await async.mapSeries(
      objectIDs,
      async (curie: string) => {
        const copyRecord = {
          ...frozenRecord,
          object: {
            original: curie,
            apiLabel: outputName,
          },
        };
        return new Record(
          copyRecord,
          this.config,
          this.edge.association,
          this.edge.reasoner_edge,
        );
      },
    );
    return transformedRecords;
  }

  /**
   * Main function to transform API response
   */
  async transform() {
    const transformedRecords = [];
    const responses = await this.pairCurieWithAPIResponse();

    await async.eachSeries(
      Object.entries(responses),
      async ([curie, curieResponses]) => {
        if (Array.isArray(curieResponses) && curieResponses.length > 0) {
          await async.eachSeries(curieResponses, async response => {
            const predicateResponse = this.jsonTransform(
              await this.wrap(response),
            );
            await async.eachSeries(
              Object.entries(predicateResponse),
              async ([_predicate, mappedResponses]) => {
                if (
                  Array.isArray(mappedResponses) &&
                  mappedResponses.length > 0
                ) {
                  await async.eachSeries(
                    mappedResponses,
                    async (mappedResponse: any[]) => {
                      transformedRecords.push(
                        ...(await this.formatRecords(curie, mappedResponse)),
                      );
                    },
                  );
                } else {
                  transformedRecords.push(
                    ...(await this.formatRecords(curie, mappedResponses)),
                  );
                }
              },
            );
          });
        }
      },
    );
    return transformedRecords;
  }

  /**
   * Retrieve all output IDs.
   * @param {Object} mappedResponse - JSON response representing an output.
   */
  extractObjectIDs(mappedResponse: object) {
    const output_id_type = this.edge.association.output_id as string;
    if (!(output_id_type in mappedResponse)) {
      return [];
    }
    mappedResponse[output_id_type] = toArray(mappedResponse[output_id_type]);
    return mappedResponse[output_id_type].map((id: string) =>
      generateCurie(output_id_type, id),
    );
  }
}
