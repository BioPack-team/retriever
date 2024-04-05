import axios, { AxiosStatic } from 'axios';
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<AxiosStatic>;

import TRAPIQueryHandler from '../../src/index';
import path from 'path';

describe('Testing TRAPIQueryHandler Module', () => {
  const disease_entity_node = {
    categories: ['Disease'],
    ids: ['MONDO:0005737'],
  };
  const gene_class_node = {
    categories: ['Gene'],
  };
  const OneHopQuery = {
    nodes: {
      n0: disease_entity_node,
      n1: gene_class_node,
    },
    edges: {
      e01: {
        subject: 'n0',
        object: 'n1',
      },
    },
  };
  describe('Testing query function', () => {
    test('test with one query edge', async () => {
      (mockedAxios.get as jest.Mock).mockResolvedValue({
        data: {
          message: {
            query_graph: {
              nodes: {
                n0: { ids: ['MONDO:0005737'], categories: ['biolink:Disease'], constraints: [] },
                n1: { ids: null, categories: ['biolink:Gene'], constraints: [] },
              },
              edges: {
                e01: {
                  predicates: ['biolink:gene_associated_with_condition'],
                  subject: 'n0',
                  object: 'n1',
                  constraints: [],
                },
              },
            },
            knowledge_graph: { nodes: {}, edges: {} },
            results: [],
          },
          max_results: 10,
          trapi_version: '1.2',
          biolink_version: '2.2.3',
          logs: [
            {
              timestamp: '2023-02-09T01:23:00.018929',
              level: 'INFO',
              message: 'Normalized curie: MONDO:0005737 to UMLS:C0282687',
              code: null,
            },
            {
              timestamp: '2023-02-09T01:23:00.192123',
              level: 'INFO',
              message: 'Converted category biolink:Disease to biolink:Disease using Biolink semantic operations.',
              code: null,
            },
            {
              timestamp: '2023-02-09T01:23:00.192135',
              level: 'INFO',
              message: 'Converted category biolink:Gene to biolink:Gene using Biolink semantic operations.',
              code: null,
            },
            {
              timestamp: '2023-02-09T01:23:00.192159',
              level: 'WARNING',
              message:
                'Biolink predicate biolink:gene_associated_with_condition is not inherently supported and could not find any supported descendants,',
              code: null,
            },
            {
              timestamp: '2023-02-09T01:23:00.198081',
              level: 'ERROR',
              message: 'Predicate: biolink:gene_associated_with_condition not supported in our meta knowledge graph.',
              code: null,
            },
            {
              timestamp: '2023-02-09T01:23:00.380552',
              level: 'INFO',
              message: 'Disease with curie UMLS:C0282687 does not exist in the database.',
              code: null,
            },
            {
              timestamp: '2023-02-09T01:23:00.380980',
              level: 'INFO',
              message: 'Disease with curie UMLS:C0282687 does not exist in the database.',
              code: null,
            },
            { timestamp: '2023-02-09T01:23:00.386890', level: 'INFO', message: 'No results found.', code: null },
          ],
          id: '98690125-8fc4-4dce-8056-18add11f58ed',
          status: 'Success',
          description: null,
          workflow: [{ id: 'lookup' }],
        },
      });
      (mockedAxios.post as jest.Mock).mockResolvedValue({
        data: {
          'MONDO:0005737': {
            id: { identifier: 'MONDO:0005737', label: 'Ebola hemorrhagic fever' },
            equivalent_identifiers: [
              { identifier: 'MONDO:0005737', label: 'Ebola hemorrhagic fever' },
              { identifier: 'DOID:4325', label: 'Ebola hemorrhagic fever' },
              { identifier: 'ORPHANET:319218' },
              { identifier: 'UMLS:C0282687', label: 'Hemorrhagic Fever, Ebola' },
              { identifier: 'MESH:D019142', label: 'Hemorrhagic Fever, Ebola' },
              { identifier: 'MEDDRA:10014071' },
              { identifier: 'MEDDRA:10014072' },
              { identifier: 'MEDDRA:10014074' },
              { identifier: 'MEDDRA:10055245' },
              { identifier: 'NCIT:C36171', label: 'Ebola Hemorrhagic Fever' },
              { identifier: 'SNOMEDCT:37109004' },
              { identifier: 'ICD10:A98.4' },
            ],
            type: [
              'biolink:Disease',
              'biolink:DiseaseOrPhenotypicFeature',
              'biolink:ThingWithTaxon',
              'biolink:BiologicalEntity',
              'biolink:NamedThing',
              'biolink:Entity',
            ],
            information_content: 100,
          },
        },
      });
      const queryHandler = new TRAPIQueryHandler(
        {},
        path.resolve(__dirname, '../../../bte-server/data/smartapi_specs.json'),
        path.resolve(__dirname, '../../../bte-server/data/predicates.json'),
      );
      queryHandler.setQueryGraph(OneHopQuery);
      await queryHandler.query();
      expect(queryHandler.knowledgeGraph.kg).toHaveProperty('nodes');
    }, 30000);
  });
});
