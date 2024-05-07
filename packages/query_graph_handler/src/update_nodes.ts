import { resolveSRI } from 'biomedical_id_resolver';
import Debug from 'debug';
import { ResolverInput, SRIResolverOutput } from 'biomedical_id_resolver';
import { Record, QEdge } from '@retriever/graph';
import { SRIBioEntity } from 'biomedical_id_resolver';
const debug = Debug('retriever:nodeUpdateHandler');

export interface CuriesByCategory {
  [category: string]: string[];
}

export default class NodesUpdateHandler {
  qEdges: QEdge[];
  constructor(qEdges: QEdge[]) {
    this.qEdges = qEdges;
  }

  /**
   * @private
   */
  _getCuries(qEdges: QEdge[]): CuriesByCategory {
    const curies: CuriesByCategory = {};
    qEdges.map((qEdge) => {
      if (qEdge.hasInput()) {
        const inputCategories = qEdge.getInputNode().getCategories();
        inputCategories.map((category) => {
          if (!(category in curies)) {
            curies[category] = [];
          }
          curies[category] = [...curies[category], ...qEdge.getInputCurie()];
        });
      }
    });
    return curies;
  }

  /**
   * Resolve input ids
   * @param {object} curies - each key represents the category, e.g. gene, value is an array of curies.
   */
  async _getEquivalentIDs(curies: ResolverInput): Promise<SRIResolverOutput> {
    // const resolver = new id_resolver.Resolver('biolink');
    // const equivalentIDs = await resolver.resolve(curies);
    return await resolveSRI(curies);
  }

  async setEquivalentIDs(qEdges: QEdge[]): Promise<void> {
    debug(`Getting equivalent IDs...`);
    const curies = this._getCuries(this.qEdges);
    debug(`curies: ${JSON.stringify(curies)}`);
    const equivalentIDs = await this._getEquivalentIDs(curies);
    qEdges.map((qEdge) => {
      const edgeEquivalentIDs = Object.keys(equivalentIDs)
        .filter((key) => qEdge.getInputCurie().includes(key))
        .reduce((res, key) => {
          return { ...res, [key]: equivalentIDs[key] };
        }, {});
      debug(`Got Edge Equivalent IDs successfully.`);
      if (Object.keys(edgeEquivalentIDs).length > 0) {
        qEdge.getInputNode().setEquivalentIDs(edgeEquivalentIDs);
      }
    });
    return;
  }

  _createEquivalentIDsObject(record: Record): { [curie: string]: SRIBioEntity } {
    if (record.object.normalizedInfo !== undefined) {
      return {
        [record.object.curie]: record.object.normalizedInfo,
      };
    } else {
      return;
    }
  }

  /**
   * Update nodes with equivalent ids based on query response.
   * @param {object} queryRecords - query response
   */
  update(queryRecords: Record[]): void {
    // queryRecords.map(record => {
    //     record.$edge_metadata.trapi_qEdge_obj.getOutputNode().updateEquivalentIDs(
    //         this._createEquivalentIDsObject(record)
    //     );
    // })
    queryRecords.map((record) => {
      if (record && !(record.object.curie in record.qEdge.getOutputNode().getEquivalentIDs())) {
        record.qEdge.getOutputNode().updateEquivalentIDs({ [record.object.curie]: record.object.normalizedInfo });
      }
    });
  }
}
