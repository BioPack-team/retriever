import BaseTransformer from "./transformer";
import { Record } from "../record";
import { JSONDoc } from "../json_transform/types";
import { removeBioLinkPrefix } from "@biothings-explorer/utils";

export default class TRAPITransformer extends BaseTransformer {
  _getUniqueEdges() {
    const edges = {};
    if (
      "message" in this.data.response &&
      "results" in this.data.response.message &&
      Array.isArray(this.data.response.message.results)
    ) {
      this.data.response.message.results.forEach(result => {
        result.analyses.forEach(analysis => {
          analysis?.edge_bindings?.e01?.forEach(binding => {
            const edgeID = binding?.id;
            const edge =
              "message" in this.data.response && edgeID
                ? this.data.response.message.knowledge_graph.edges[edgeID]
                : undefined;
            const edgeHasSupportGraph = edge.attributes.some(attribute => {
              if (
                attribute.attribute_type_id === "biolink:support_graphs" &&
                attribute.value?.length
              ) {
                return true;
              }
            });
            if (edgeHasSupportGraph || !edgeID) return;
            edges[edgeID] = {
              subject:
                (this.data.response as JSONDoc).message.knowledge_graph.edges[
                  edgeID
                ].subject ?? result.node_bindings.n0[0].id,
              object:
                (this.data.response as JSONDoc).message.knowledge_graph.edges[
                  edgeID
                ].object ?? result.node_bindings.n1[0].id,
            };
          });
        });
      });
    }
    return edges;
  }

  _getEdgeInfo(edgeID) {
    if (
      "message" in this.data.response &&
      "knowledge_graph" in this.data.response.message &&
      "edges" in this.data.response.message.knowledge_graph
    ) {
      return this.data.response.message.knowledge_graph.edges[edgeID];
    }
    return undefined;
  }

  _transformIndividualEdge(edge, edgeBinding) {
    const frozenRecord = {
      subject: {
        ...this._getSubject(edgeBinding.subject),
        apiLabel: undefined,
      },
      object: {
        original: edgeBinding.object,
        apiLabel: undefined, // could get from API
      },
      qualifiers: edge.qualifiers
        ? Object.fromEntries(
            edge.qualifiers.map(qualifier => {
              return [qualifier.qualifier_type_id, qualifier.qualifier_value];
            }),
          )
        : undefined,
      association: this.edge.association,
      qEdge: this.edge.reasoner_edge,
      mappedResponse: {
        "edge-attributes": [...edge.attributes],
        trapi_sources: edge.sources,
      },
    };
    const hasOriginal = !!frozenRecord.subject.original;
    const predicateMatches =
      removeBioLinkPrefix(this.edge.association.predicate) ==
      removeBioLinkPrefix(edge.predicate);
    if (hasOriginal && predicateMatches) {
      return new Record(
        frozenRecord,
        this.config,
        this.edge.association,
        this.edge.reasoner_edge,
      );
    }
  }

  async transform() {
    const edgeBindings = this._getUniqueEdges();
    return Object.keys(edgeBindings).map(edge => {
      const edgeInfo = this._getEdgeInfo(edge);
      if (!(typeof edgeInfo === "undefined")) {
        return this._transformIndividualEdge(edgeInfo, edgeBindings[edge]);
      }
    });
  }
}
