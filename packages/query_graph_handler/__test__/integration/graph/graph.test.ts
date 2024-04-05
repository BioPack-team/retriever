import graph from '../../../src/graph/graph';
import { Record } from '@biothings-explorer/api-response-transform';

describe('Test graph class', () => {
  const qNode1 = {
    getID() {
      return 'qg1';
    },
  };
  const qNode2 = {
    getID() {
      return 'qg2';
    },
  };
  const record1 = new Record({
    api: 'API1',
    metaEdgeSource: 'source1',
    apiInforesCurie: 'infores:API1',
    predicate: 'predicate1',
    object: {
      qNodeID: 'qg2',
      curie: 'outputPrimaryCurie',
      original: 'outputPrimaryCurie',
    },
    subject: {
      qNodeID: 'qg1',
      curie: 'inputPrimaryCurie',
      original: 'inputPrimaryCurie',
    },
    publications: ['PMID:1', 'PMID:2'],
    mappedResponse: {
      relation: 'relation1',
    },
  });

  const record2 = new Record({
    api: 'API2',
    metaEdgeSource: 'source2',
    apiInforesCurie: 'infores:API2',
    predicate: 'predicate1',
    object: {
      qNodeID: 'qg2',
      curie: 'outputPrimaryCurie',
      original: 'outputPrimaryCurie',
    },
    subject: {
      qNodeID: 'qg1',
      curie: 'inputPrimaryCurie',
      original: 'inputPrimaryCurie',
    },
    publications: ['PMC:1', 'PMC:2'],
    mappedResponse: {
      relation: 'relation2',
    },
  });

  const record3 = new Record({
    api: 'API3',
    metaEdgeSource: 'source3',
    apiInforesCurie: 'infores:API3',
    predicate: 'predicate2',
    object: {
      qNodeID: 'qg2',
      curie: 'outputPrimaryCurie',
      original: 'outputPrimaryCurie',
    },
    subject: {
      qNodeID: 'qg1',
      curie: 'inputPrimaryCurie',
      original: 'inputPrimaryCurie',
    },
    publications: ['PMC:3', 'PMC:4'],
    mappedResponse: {
      relation: 'relation3',
    },
  });

  const record3a = new Record({
    api: 'API3',
    metaEdgeSource: 'source3',
    apiInforesCurie: 'infores:API3',
    predicate: 'predicate2',
    object: {
      qNodeID: 'qg2',
      curie: 'outputPrimaryCurie',
      original: 'outputPrimaryCurie',
    },
    subject: {
      qNodeID: 'qg1',
      curie: 'inputPrimaryCurie',
      original: 'inputPrimaryCurie',
    },
    publications: ['PMC:6', 'PMC:7'],
    mappedResponse: {
      relation: ['relation3a', 'relation3b'],
    },
  });

  test('A single query result is correctly updated.', () => {
    const g = new graph();
    g.update([record1]);
    expect(g.nodes).toHaveProperty('outputPrimaryCurie');
    expect(g.nodes).toHaveProperty('inputPrimaryCurie');
    expect(g.nodes['outputPrimaryCurie'].primaryCurie).toEqual('outputPrimaryCurie');
    expect(g.nodes['outputPrimaryCurie'].qNodeID).toEqual('qg2');
    expect(Array.from(g.nodes['outputPrimaryCurie'].sourceNodes)).toEqual(['inputPrimaryCurie']);
    expect(Array.from(g.nodes['outputPrimaryCurie'].sourceQNodeIDs)).toEqual(['qg1']);
    expect(g.nodes['inputPrimaryCurie'].primaryCurie).toEqual('inputPrimaryCurie');
    expect(g.nodes['inputPrimaryCurie'].qNodeID).toEqual('qg1');
    expect(Array.from(g.nodes['inputPrimaryCurie'].targetNodes)).toEqual(['outputPrimaryCurie']);
    expect(Array.from(g.nodes['inputPrimaryCurie'].targetQNodeIDs)).toEqual(['qg2']);
    expect(g.edges).toHaveProperty('95fe2a8089c0d79ea093b97c5991f7ba');
    expect(Array.from(g.edges['95fe2a8089c0d79ea093b97c5991f7ba'].apis)).toEqual(['API1']);
    expect(g.edges['95fe2a8089c0d79ea093b97c5991f7ba'].sources).toHaveProperty('source1');
    expect(Array.from(g.edges['95fe2a8089c0d79ea093b97c5991f7ba'].publications)).toEqual(['PMID:1', 'PMID:2']);
    expect(g.edges['95fe2a8089c0d79ea093b97c5991f7ba'].attributes).toHaveProperty('relation', new Set(['relation1']));
  });

  test('Multiple query results are correctly updated for two edges having same input, predicate and output', () => {
    const g = new graph();
    g.update([record1, record2]);
    expect(g.nodes).toHaveProperty('outputPrimaryCurie');
    expect(g.nodes).toHaveProperty('inputPrimaryCurie');
    expect(g.nodes['outputPrimaryCurie'].primaryCurie).toEqual('outputPrimaryCurie');
    expect(g.nodes['outputPrimaryCurie'].qNodeID).toEqual('qg2');
    expect(Array.from(g.nodes['outputPrimaryCurie'].sourceNodes)).toEqual(['inputPrimaryCurie']);
    expect(Array.from(g.nodes['outputPrimaryCurie'].sourceQNodeIDs)).toEqual(['qg1']);
    expect(g.nodes['inputPrimaryCurie'].primaryCurie).toEqual('inputPrimaryCurie');
    expect(g.nodes['inputPrimaryCurie'].qNodeID).toEqual('qg1');
    expect(Array.from(g.nodes['inputPrimaryCurie'].targetNodes)).toEqual(['outputPrimaryCurie']);
    expect(Array.from(g.nodes['inputPrimaryCurie'].targetQNodeIDs)).toEqual(['qg2']);

    expect(g.edges).toHaveProperty('95fe2a8089c0d79ea093b97c5991f7ba');
    expect(Array.from(g.edges['95fe2a8089c0d79ea093b97c5991f7ba'].apis)).toEqual(['API1']);
    expect(g.edges['95fe2a8089c0d79ea093b97c5991f7ba'].sources).toHaveProperty('source1');
    expect(Array.from(g.edges['95fe2a8089c0d79ea093b97c5991f7ba'].publications)).toEqual(['PMID:1', 'PMID:2']);
    expect(g.edges['95fe2a8089c0d79ea093b97c5991f7ba'].attributes).toHaveProperty('relation', new Set(['relation1']));

    expect(g.edges).toHaveProperty('9d334cb674d5671364c45cc8403184c6');
    expect(Array.from(g.edges['9d334cb674d5671364c45cc8403184c6'].apis)).toEqual(['API2']);
    expect(g.edges['9d334cb674d5671364c45cc8403184c6'].sources).toHaveProperty('source2');
    expect(Array.from(g.edges['9d334cb674d5671364c45cc8403184c6'].publications)).toEqual(['PMC:1', 'PMC:2']);
    expect(g.edges['9d334cb674d5671364c45cc8403184c6'].attributes).toHaveProperty('relation', new Set(['relation2']));
  });

  test('Multiple query results for different edges are correctly updated', () => {
    const g = new graph();
    g.update([record1, record2, record3]);
    expect(g.nodes).toHaveProperty('outputPrimaryCurie');
    expect(g.nodes).toHaveProperty('inputPrimaryCurie');
    expect(g.nodes['outputPrimaryCurie'].primaryCurie).toEqual('outputPrimaryCurie');
    expect(g.nodes['outputPrimaryCurie'].qNodeID).toEqual('qg2');
    expect(Array.from(g.nodes['outputPrimaryCurie'].sourceNodes)).toEqual(['inputPrimaryCurie']);
    expect(Array.from(g.nodes['outputPrimaryCurie'].sourceQNodeIDs)).toEqual(['qg1']);
    expect(g.nodes['inputPrimaryCurie'].primaryCurie).toEqual('inputPrimaryCurie');
    expect(g.nodes['inputPrimaryCurie'].qNodeID).toEqual('qg1');
    expect(Array.from(g.nodes['inputPrimaryCurie'].targetNodes)).toEqual(['outputPrimaryCurie']);
    expect(Array.from(g.nodes['inputPrimaryCurie'].targetQNodeIDs)).toEqual(['qg2']);

    expect(g.edges).toHaveProperty('95fe2a8089c0d79ea093b97c5991f7ba');
    expect(Array.from(g.edges['95fe2a8089c0d79ea093b97c5991f7ba'].apis)).toEqual(['API1']);
    expect(g.edges['95fe2a8089c0d79ea093b97c5991f7ba'].sources).toHaveProperty('source1');
    expect(Array.from(g.edges['95fe2a8089c0d79ea093b97c5991f7ba'].publications)).toEqual(['PMID:1', 'PMID:2']);
    expect(g.edges['95fe2a8089c0d79ea093b97c5991f7ba'].attributes).toHaveProperty('relation', new Set(['relation1']));

    expect(g.edges).toHaveProperty('9d334cb674d5671364c45cc8403184c6');
    expect(Array.from(g.edges['9d334cb674d5671364c45cc8403184c6'].apis)).toEqual(['API2']);
    expect(g.edges['9d334cb674d5671364c45cc8403184c6'].sources).toHaveProperty('source2');
    expect(Array.from(g.edges['9d334cb674d5671364c45cc8403184c6'].publications)).toEqual(['PMC:1', 'PMC:2']);
    expect(g.edges['9d334cb674d5671364c45cc8403184c6'].attributes).toHaveProperty('relation', new Set(['relation2']));

    expect(g.edges).toHaveProperty('4fe2d5d3e03e0f78f272745caf6b627d');
    expect(Array.from(g.edges['4fe2d5d3e03e0f78f272745caf6b627d'].apis)).toEqual(['API3']);
    expect(g.edges['4fe2d5d3e03e0f78f272745caf6b627d'].sources).toHaveProperty('source3');
    expect(Array.from(g.edges['4fe2d5d3e03e0f78f272745caf6b627d'].publications)).toEqual(['PMC:3', 'PMC:4']);
    expect(g.edges['4fe2d5d3e03e0f78f272745caf6b627d'].attributes).toHaveProperty('relation', new Set(['relation3']));
  });

  test('Multiple attributes with the same name are merged', () => {
    const g = new graph();
    g.update([record3, record3a]);

    expect(g.edges).toHaveProperty('4fe2d5d3e03e0f78f272745caf6b627d');
    expect(Array.from(g.edges['4fe2d5d3e03e0f78f272745caf6b627d'].publications)).toEqual([
      'PMC:3',
      'PMC:4',
      'PMC:6',
      'PMC:7',
    ]);
    expect(g.edges['4fe2d5d3e03e0f78f272745caf6b627d'].attributes).toHaveProperty(
      'relation',
      new Set(['relation3', 'relation3a', 'relation3b']),
    );
  });
});
