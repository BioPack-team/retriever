import { cloneDeep, range } from 'lodash';
import QNode from '../../src/query_node';
import QEdge from '../../src/query_edge';
import QueryResult from '../../src/results_assembly/query_results';
import { Record } from '@biothings-explorer/api-response-transform';
import { EDGE_ATTRIBUTES_USED_IN_RECORD_HASH } from '../../src/config';

describe('Testing QueryResults Module', () => {
  describe('"Real" Records', () => {
    describe('Single Record', () => {
      const gene_node1 = new QNode({ id: 'n1', categories: ['Gene'], ids: ['NCBIGene:632'] });
      const chemical_node1 = new QNode({ id: 'n2', categories: ['ChemicalSubstance'] });
      const edge1 = new QEdge({ id: 'e01', subject: gene_node1, object: chemical_node1 });
      const record = new Record(
        {
          publications: ['PMID:8366144', 'PMID:8381250'],
          mappedResponse: {
            relation: 'antagonist',
            source: 'DrugBank',
            score: '0.9',
          },
          //@ts-expect-error: partial data for specific test
          subject: {
            original: 'SYMBOL:BGLAP',
            normalizedInfo: {
              primaryID: 'NCBIGene:632',
              label: 'BGLAP',
              equivalentIDs: ['SYMBOL:BGLAP', 'NCBIGene:632'],
            },
          },
          //@ts-expect-error: partial data for specific test
          object: {
            original: 'CHEMBL.COMPOUND:CHEMBL1200983',
            normalizedInfo: {
              primaryID: 'CHEMBL.COMPOUND:CHEMBL1200983',
              label: 'GALLIUM NITRATE',
              equivalentIDs: ['CHEMBL.COMPOUND:CHEMBL1200983', 'PUBCHEM.COMPOUND:5282394', 'name:GALLIUM NITRATE'],
            },
          },
        },
        EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
        {
          predicate: 'biolink:physically_interacts_with',
          source: 'DGIdb',
          api_name: 'BioThings DGIDB API',
        },
        edge1,
      );

      test('should get n1, n2 and e01', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e01: {
            connected_to: [],
            records: [record],
          },
        });
        expect(queryResult.getResults().length).toEqual(1);
        expect(queryResult.getResults()[0].node_bindings).toHaveProperty('n1');
        expect(queryResult.getResults()[0].node_bindings).toHaveProperty('n2');
        expect(queryResult.getResults()[0].analyses[0].edge_bindings).toHaveProperty('e01');
        expect(queryResult.getResults()[0].analyses[0]).toHaveProperty('score');
      });
    });

    describe('Two Records', () => {
      describe('query graph: gene1-disease1-gene1', () => {
        const gene_node_start = new QNode({ id: 'n1', categories: ['Gene'] });
        const disease_node = new QNode({ id: 'n2', categories: ['Disease'] });
        const gene_node_end = new QNode({ id: 'n3', categories: ['Gene'] });

        const edge1 = new QEdge({ id: 'e01', subject: gene_node_start, object: disease_node });
        const edge2 = new QEdge({ id: 'e02', subject: disease_node, object: gene_node_end });

        const record1 = new Record(
          {
            publications: ['PMID:123', 'PMID:1234'],
            //@ts-expect-error: partial data for specific test
            subject: {
              original: 'SYMBOL:KCNMA1',
              normalizedInfo: {
                primaryID: 'NCBIGene:3778',
                label: 'KCNMA1',
                equivalentIDs: ['SYMBOL:KCNMA1', 'NCBIGene:3778'],
              },
            },
            //@ts-expect-error: partial data for specific test
            object: {
              original: 'MONDO:0011122',
              normalizedInfo: {
                primaryID: 'MONDO:0011122',
                label: 'obesity disorder',
                equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
              },
            },
          },
          EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
          {
            predicate: 'biolink:gene_associated_with_condition',
            api_name: 'Automat Pharos',
          },
          edge1,
        );

        const record2 = new Record(
          {
            publications: ['PMID:345', 'PMID:456'],
            //@ts-expect-error: partial data for specific test
            subject: {
              original: 'MONDO:0011122',
              normalizedInfo: {
                primaryID: 'MONDO:0011122',
                label: 'obesity disorder',
                equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
              },
            },
            //@ts-expect-error: partial data for specific test
            object: {
              original: 'SYMBOL:KCNMA1',
              normalizedInfo: {
                primaryID: 'NCBIGene:3778',
                label: 'KCNMA1',
                equivalentIDs: ['SYMBOL:KCNMA1', 'NCBIGene:3778'],
              },
            },
          },
          EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
          {
            predicate: 'biolink:condition_associated_with_gene',
            api_name: 'Automat Hetio',
          },
          edge2,
        );

        test('should get n1, n2, n3 and e01, e02', async () => {
          const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });

          await queryResult.update({
            e01: {
              connected_to: ['e02'],
              records: [record1],
            },
            e02: {
              connected_to: ['e01'],
              records: [record2],
            },
          });

          const results = queryResult.getResults();

          expect(results.length).toEqual(1);

          expect(Object.keys(results[0].node_bindings).length).toEqual(3);
          expect(results[0].node_bindings).toHaveProperty('n1');
          expect(results[0].node_bindings).toHaveProperty('n2');
          expect(results[0].node_bindings).toHaveProperty('n3');

          expect(Object.keys(results[0].analyses[0].edge_bindings).length).toEqual(2);
          expect(results[0].analyses[0].edge_bindings).toHaveProperty('e01');
          expect(results[0].analyses[0].edge_bindings).toHaveProperty('e02');

          expect(results[0].analyses[0]).toHaveProperty('score');
        });
      });

      describe('query graph: gene1-disease1-gene2 (no ids params)', () => {
        const gene_node_start = new QNode({ id: 'n1', categories: ['Gene'] });
        const disease_node = new QNode({ id: 'n2', categories: ['Disease'] });
        const gene_node_end = new QNode({ id: 'n3', categories: ['Gene'] });

        const edge1 = new QEdge({ id: 'e01', subject: gene_node_start, object: disease_node });
        const edge2 = new QEdge({ id: 'e02', subject: disease_node, object: gene_node_end });

        const record1 = new Record(
          {
            publications: ['PMID:123', 'PMID:1234'],
            //@ts-expect-error: partial data for specific test
            subject: {
              original: 'SYMBOL:KCNMA1',
              normalizedInfo: {
                primaryID: 'NCBIGene:3778',
                label: 'KCNMA1',
                equivalentIDs: ['SYMBOL:KCNMA1', 'NCBIGene:3778'],
              },
            },
            //@ts-expect-error: partial data for specific test
            object: {
              original: 'MONDO:0011122',
              normalizedInfo: {
                primaryID: 'MONDO:0011122',
                label: 'obesity disorder',
                equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
              },
            },
          },
          EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
          {
            predicate: 'biolink:gene_associated_with_condition',
            api_name: 'Automat Pharos',
          },
          edge1,
        );

        const record2 = new Record(
          {
            publications: ['PMID:345', 'PMID:456'],
            //@ts-expect-error: partial data for specific test
            subject: {
              original: 'MONDO:0011122',
              normalizedInfo: {
                primaryID: 'MONDO:0011122',
                label: 'obesity disorder',
                equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
              },
            },
            //@ts-expect-error: partial data for specific test
            object: {
              original: 'SYMBOL:TULP3',
              normalizedInfo: {
                primaryID: 'NCBIGene:7289',
                label: 'TULP3',
                equivalentIDs: ['SYMBOL:TULP3', 'NCBIGene:7289'],
              },
            },
          },
          EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
          {
            predicate: 'biolink:condition_associated_with_gene',
            api_name: 'Automat Hetio',
          },
          edge2,
        );

        test('should get n1, n2, n3 and e01, e02', async () => {
          const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
          await queryResult.update({
            e01: {
              connected_to: ['e02'],
              records: [record1],
            },
            e02: {
              connected_to: ['e01'],
              records: [record2],
            },
          });

          const results = queryResult.getResults();

          expect(results.length).toEqual(1);

          expect(Object.keys(results[0].node_bindings).length).toEqual(3);
          expect(results[0].node_bindings).toHaveProperty('n1');
          expect(results[0].node_bindings).toHaveProperty('n2');
          expect(results[0].node_bindings).toHaveProperty('n3');

          expect(Object.keys(results[0].analyses[0].edge_bindings).length).toEqual(2);
          expect(results[0].analyses[0].edge_bindings).toHaveProperty('e01');
          expect(results[0].analyses[0].edge_bindings).toHaveProperty('e02');

          expect(results[0].analyses[0]).toHaveProperty('score');
        });
      });

      describe('query graph: gene1-disease1-gene2 (gene1 has ids param)', () => {
        const gene_node_start = new QNode({ id: 'n1', categories: ['Gene'], ids: ['NCBIGene:3778'] });
        const disease_node = new QNode({ id: 'n2', categories: ['Disease'] });
        const gene_node_end = new QNode({ id: 'n3', categories: ['Gene'] });

        const edge1 = new QEdge({ id: 'e01', subject: gene_node_start, object: disease_node });
        const edge2 = new QEdge({ id: 'e02', subject: disease_node, object: gene_node_end });

        const record1 = new Record(
          {
            publications: ['PMID:123', 'PMID:1234'],
            //@ts-expect-error: partial data for specific test
            subject: {
              original: 'SYMBOL:KCNMA1',
              normalizedInfo: {
                primaryID: 'NCBIGene:3778',
                label: 'KCNMA1',
                equivalentIDs: ['SYMBOL:KCNMA1', 'NCBIGene:3778'],
              },
            },
            //@ts-expect-error: partial data for specific test
            object: {
              original: 'MONDO:0011122',
              normalizedInfo: {
                primaryID: 'MONDO:0011122',
                label: 'obesity disorder',
                equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
              },
            },
          },
          EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
          {
            predicate: 'biolink:gene_associated_with_condition',
            api_name: 'Automat Pharos',
          },
          edge1,
        );

        const record2 = new Record(
          {
            publications: ['PMID:345', 'PMID:456'],
            //@ts-expect-error: partial data for specific test
            subject: {
              original: 'MONDO:0011122',
              normalizedInfo: {
                primaryID: 'MONDO:0011122',
                label: 'obesity disorder',
                equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
              },
            },
            //@ts-expect-error: partial data for specific test
            object: {
              original: 'SYMBOL:TULP3',
              normalizedInfo: {
                primaryID: 'NCBIGene:7289',
                label: 'TULP3',
                equivalentIDs: ['SYMBOL:TULP3', 'NCBIGene:7289'],
              },
            },
          },
          EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
          {
            predicate: 'biolink:condition_associated_with_gene',
            api_name: 'Automat Hetio',
          },
          edge2,
        );

        test('should get n1, n2, n3 and e01, e02', async () => {
          const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
          await queryResult.update({
            e01: {
              connected_to: ['e02'],
              records: [record1],
            },
            e02: {
              connected_to: ['e01'],
              records: [record2],
            },
          });

          const results = queryResult.getResults();

          expect(results.length).toEqual(1);

          expect(Object.keys(results[0].node_bindings).length).toEqual(3);
          expect(results[0].node_bindings).toHaveProperty('n1');
          expect(results[0].node_bindings).toHaveProperty('n2');
          expect(results[0].node_bindings).toHaveProperty('n3');

          expect(Object.keys(results[0].analyses[0].edge_bindings).length).toEqual(2);
          expect(results[0].analyses[0].edge_bindings).toHaveProperty('e01');
          expect(results[0].analyses[0].edge_bindings).toHaveProperty('e02');

          expect(results[0].analyses[0]).toHaveProperty('score');
        });
      });

      describe('query graph: gene1-disease1-gene2 (gene1 & gene2 have ids params)', () => {
        const gene_node_start = new QNode({ id: 'n1', categories: ['Gene'], ids: ['NCBIGene:3778'] });
        const disease_node = new QNode({ id: 'n2', categories: ['Disease'] });
        const gene_node_end = new QNode({ id: 'n3', categories: ['Gene'], ids: ['NCBIGene:7289'] });

        const edge1 = new QEdge({ id: 'e01', subject: gene_node_start, object: disease_node });
        const edge2 = new QEdge({ id: 'e02', subject: disease_node, object: gene_node_end });

        const record1 = new Record(
          {
            publications: ['PMID:123', 'PMID:1234'],
            //@ts-expect-error: partial data for specific test
            subject: {
              original: 'SYMBOL:KCNMA1',
              normalizedInfo: {
                primaryID: 'NCBIGene:3778',
                label: 'KCNMA1',
                equivalentIDs: ['SYMBOL:KCNMA1', 'NCBIGene:3778'],
              },
            },
            //@ts-expect-error: partial data for specific test
            object: {
              original: 'MONDO:0011122',
              normalizedInfo: {
                primaryID: 'MONDO:0011122',
                label: 'obesity disorder',
                equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
              },
            },
          },
          EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
          {
            predicate: 'biolink:gene_associated_with_condition',
            api_name: 'Automat Pharos',
          },
          edge1,
        );

        // NOTE: I had to switch subject and object.
        // Compare with first test of this type.
        const record2 = new Record(
          {
            publications: ['PMID:345', 'PMID:456'],
            //@ts-expect-error: partial data for specific test
            subject: {
              original: 'SYMBOL:TULP3',
              normalizedInfo: {
                primaryID: 'NCBIGene:7289',
                label: 'TULP3',
                equivalentIDs: ['SYMBOL:TULP3', 'NCBIGene:7289'],
              },
            },
            //@ts-expect-error: partial data for specific test
            object: {
              original: 'MONDO:0011122',
              normalizedInfo: {
                primaryID: 'MONDO:0011122',
                label: 'obesity disorder',
                equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
              },
            },
          },
          EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
          {
            predicate: 'biolink:condition_associated_with_gene',
            api_name: 'Automat Hetio',
          },
          edge2,
        );

        test('should get n1, n2, n3 and e01, e02', async () => {
          const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
          await queryResult.update({
            e01: {
              connected_to: ['e02'],
              records: [record1],
            },
            e02: {
              connected_to: ['e01'],
              records: [record2],
            },
          });

          const results = queryResult.getResults();

          expect(results.length).toEqual(1);

          expect(Object.keys(results[0].node_bindings).length).toEqual(3);
          expect(results[0].node_bindings).toHaveProperty('n1');
          expect(results[0].node_bindings).toHaveProperty('n2');
          expect(results[0].node_bindings).toHaveProperty('n3');

          expect(Object.keys(results[0].analyses[0].edge_bindings).length).toEqual(2);
          expect(results[0].analyses[0].edge_bindings).toHaveProperty('e01');
          expect(results[0].analyses[0].edge_bindings).toHaveProperty('e02');

          expect(results[0].analyses[0]).toHaveProperty('score');
        });
      });

      describe('query graph: gene1-disease1-gene2 (gene2 has ids param)', () => {
        const gene_node_start = new QNode({ id: 'n1', categories: ['Gene'] });
        const disease_node = new QNode({ id: 'n2', categories: ['Disease'] });
        const gene_node_end = new QNode({ id: 'n3', categories: ['Gene'], ids: ['NCBIGene:7289'] });

        const edge1 = new QEdge({ id: 'e01', subject: gene_node_start, object: disease_node });
        const edge2 = new QEdge({ id: 'e02', subject: disease_node, object: gene_node_end });

        const record1 = new Record(
          {
            publications: ['PMID:123', 'PMID:1234'],
            //@ts-expect-error: partial data for specific test
            subject: {
              original: 'SYMBOL:KCNMA1',
              normalizedInfo: {
                primaryID: 'NCBIGene:3778',
                label: 'KCNMA1',
                equivalentIDs: ['SYMBOL:KCNMA1', 'NCBIGene:3778'],
              },
            },
            //@ts-expect-error: partial data for specific test
            object: {
              original: 'MONDO:0011122',
              normalizedInfo: {
                primaryID: 'MONDO:0011122',
                label: 'obesity disorder',
                equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
              },
            },
          },
          EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
          {
            predicate: 'biolink:gene_associated_with_condition',
            api_name: 'Automat Pharos',
          },
          edge1,
        );

        // NOTE: I had to switch subject and object.
        // Compare with first test of this type.
        const record2 = new Record(
          {
            publications: ['PMID:345', 'PMID:456'],
            //@ts-expect-error: partial data for specific test
            subject: {
              original: 'SYMBOL:TULP3',
              normalizedInfo: {
                primaryID: 'NCBIGene:7289',
                label: 'TULP3',
                equivalentIDs: ['SYMBOL:TULP3', 'NCBIGene:7289'],
              },
            },
            //@ts-expect-error: partial data for specific test
            object: {
              original: 'MONDO:0011122',
              normalizedInfo: {
                primaryID: 'MONDO:0011122',
                label: 'obesity disorder',
                equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
              },
            },
          },
          EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
          {
            predicate: 'biolink:condition_associated_with_gene',
            api_name: 'Automat Hetio',
          },
          edge2,
        );

        test('should get n1, n2, n3 and e01, e02', async () => {
          const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
          await queryResult.update({
            e01: {
              connected_to: ['e02'],
              records: [record1],
            },
            e02: {
              connected_to: ['e01'],
              records: [record2],
            },
          });

          const results = queryResult.getResults();

          expect(results.length).toEqual(1);

          expect(Object.keys(results[0].node_bindings).length).toEqual(3);
          expect(results[0].node_bindings).toHaveProperty('n1');
          expect(results[0].node_bindings).toHaveProperty('n2');
          expect(results[0].node_bindings).toHaveProperty('n3');

          expect(Object.keys(results[0].analyses[0].edge_bindings).length).toEqual(2);
          expect(results[0].analyses[0].edge_bindings).toHaveProperty('e01');
          expect(results[0].analyses[0].edge_bindings).toHaveProperty('e02');

          expect(results[0].analyses[0]).toHaveProperty('score');
        });
      });
    });

    describe('Three Records', () => {
      const gene_node_start = new QNode({ id: 'n1', categories: ['Gene'], ids: ['NCBIGene:3778'] });
      const disease_node = new QNode({ id: 'n2', categories: ['Disease'] });
      const gene_node_end = new QNode({ id: 'n3', categories: ['Gene'] });

      const edge1 = new QEdge({ id: 'e01', subject: gene_node_start, object: disease_node });
      const edge2 = new QEdge({ id: 'e02', subject: disease_node, object: gene_node_end });

      const record1 = new Record(
        {
          publications: ['PMID:123', 'PMID:1234'],
          //@ts-expect-error: partial data for specific test
          subject: {
            original: 'SYMBOL:KCNMA1',
            normalizedInfo: {
              primaryID: 'NCBIGene:3778',
              label: 'KCNMA1',
              equivalentIDs: ['SYMBOL:KCNMA1', 'NCBIGene:3778'],
            },
          },
          //@ts-expect-error: partial data for specific test
          object: {
            original: 'MONDO:0011122',
            normalizedInfo: {
              primaryID: 'MONDO:0011122',
              label: 'obesity disorder',
              equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
            },
          },
        },
        EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
        {
          predicate: 'biolink:gene_associated_with_condition',
          api_name: 'Automat Pharos',
        },
        edge1,
      );

      const record2 = new Record(
        {
          publications: ['PMID:345', 'PMID:456'],
          //@ts-expect-error: partial data for specific test
          subject: {
            original: 'MONDO:0011122',
            normalizedInfo: {
              primaryID: 'MONDO:0011122',
              label: 'obesity disorder',
              equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
            },
          },
          //@ts-expect-error: partial data for specific test
          object: {
            original: 'SYMBOL:TULP3',
            normalizedInfo: {
              primaryID: 'NCBIGene:7289',
              label: 'TULP3',
              equivalentIDs: ['SYMBOL:TULP3', 'NCBIGene:7289'],
            },
          },
        },
        EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
        {
          predicate: 'biolink:condition_associated_with_gene',
          api_name: 'Automat Hetio',
        },
        edge2,
      );

      const record3 = new Record(
        {
          publications: ['PMID:987', 'PMID:876'],
          //@ts-expect-error: partial data for specific test
          subject: {
            original: 'MONDO:0011122',
            normalizedInfo: {
              primaryID: 'MONDO:0011122',
              label: 'obesity disorder',
              equivalentIDs: ['MONDO:0011122', 'MESH:D009765', 'name:obesity disorder'],
            },
          },
          //@ts-expect-error: partial data for specific test
          object: {
            original: 'SYMBOL:TECR',
            normalizedInfo: {
              primaryID: 'NCBIGene:9524',
              label: 'TECR',
              equivalentIDs: ['SYMBOL:TECR', 'NCBIGene:9524'],
            },
          },
        },
        EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
        {
          predicate: 'biolink:condition_associated_with_gene',
          api_name: 'Automat Hetio',
        },
        edge2,
      );

      test('should get 2 results when query graph is -- and records are -<', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });

        await queryResult.update({
          e01: {
            connected_to: ['e02'],
            records: [record1],
          },
          e02: {
            connected_to: ['e01'],
            records: [record2, record3],
          },
        });

        const results = queryResult.getResults();

        expect(results.length).toEqual(2);

        expect(Object.keys(results[0].node_bindings).length).toEqual(3);
        expect(results[0].node_bindings).toHaveProperty('n1');
        expect(results[0].node_bindings).toHaveProperty('n2');
        expect(results[0].node_bindings).toHaveProperty('n3');

        expect(Object.keys(results[0].analyses[0].edge_bindings).length).toEqual(2);
        expect(results[0].analyses[0].edge_bindings).toHaveProperty('e01');
        expect(results[0].analyses[0].edge_bindings).toHaveProperty('e02');

        expect(results[0].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[1].node_bindings).length).toEqual(3);
        expect(results[1].node_bindings).toHaveProperty('n1');
        expect(results[1].node_bindings).toHaveProperty('n2');
        expect(results[1].node_bindings).toHaveProperty('n3');

        expect(Object.keys(results[1].analyses[0].edge_bindings).length).toEqual(2);
        expect(results[1].analyses[0].edge_bindings).toHaveProperty('e01');
        expect(results[1].analyses[0].edge_bindings).toHaveProperty('e02');

        expect(results[1].analyses[0]).toHaveProperty('score');
      });
    });
  });

  describe('"Synthetic" Records', () => {
    // TODO: the tests will fail if we switch the subject and object. Is there a way to format the
    // query graph and/or the records such that the tests would still pass?

    const [n0, n1, n2, n3, n4, n5] = Array(6)
      .fill(0)
      .map((s, i) => {
        return {
          a: {
            qNodeID: `n${i}`,
            curie: `n${i}a`,
            original: `n${i}a`,
          },
          b: {
            qNodeID: `n${i}`,
            curie: `n${i}b`,
            original: `n${i}b`,
          },
        };
      });
    const [source0, source1] = Array(2)
      .fill(0)
      .map((s, i) => {
        return { source: `source${i}` };
      });
    const [api0, api1] = Array(2)
      .fill(0)
      .map((s, i) => {
        return { api: `api${i}` };
      });
    const recordPred = Array(7)
      .fill(0)
      .map((s, i) => {
        return { predicate: `biolink:record${i}_pred` };
      });
    const set = (node) => {
      return {
        qNodeID: `${node.qNodeID}_with_is_set`,
        curie: node.curie,
        original: node.original,
        isSet: true,
      };
    };

    // AKA record0_n0a_n1a_pred0_api0
    const record0_n0a_n1a = {
      subject: { ...n0.a },
      object: { ...n1.a },
      predicate: 'biolink:record0_pred0',
      ...source0,
      ...api0,
    };

    const record0_n1a_n0a = {
      ...record0_n0a_n1a,
      subject: { ...n1.a },
      object: { ...n0.a },
    };

    const record0_n0a_n1a_pred0_api1 = {
      ...record0_n0a_n1a,
      ...source1,
      ...api1,
    };

    const record0_n0a_n1a_pred1_api0 = {
      ...record0_n0a_n1a,
      predicate: 'biolink:record0_pred1',
    };

    const record0_n0a_n1a_pred1_api1 = {
      ...record0_n0a_n1a,
      predicate: 'biolink:record0_pred1',
      ...source1,
      ...api1,
    };

    const record0_n0a_n1b = {
      subject: { ...n0.a },
      object: { ...n1.b },
      ...recordPred[0],
      ...source0,
      ...api0,
    };

    const record0_n0b_n1a = {
      subject: { ...n0.b },
      object: { ...n1.a },
      ...recordPred[0],
      ...source0,
      ...api0,
    };

    const record0_n0b_n1b = {
      subject: { ...n0.b },
      object: { ...n1.b },
      ...recordPred[0],
      ...source0,
      ...api0,
    };

    const record1_n1a_n2a = {
      subject: { ...n1.a },
      object: { ...n2.a },
      ...recordPred[1],
      ...source0,
      ...api0,
    };

    const record1_n2a_n1a = {
      ...record1_n1a_n2a,
      subject: { ...n2.a },
      object: { ...n1.a },
    };

    const record1_n1a_n2b = {
      subject: { ...n1.a },
      object: { ...n2.b },
      ...recordPred[1],
      ...source0,
      ...api0,
    };

    const record1_n2b_n1a = {
      ...record1_n1a_n2b,
      subject: { ...n2.b },
      object: { ...n1.a },
    };

    const record1_n1b_n2a = {
      subject: { ...n1.b },
      object: { ...n2.a },
      ...recordPred[1],
      ...source0,
      ...api0,
    };

    const record1_n1b_n2b = {
      subject: { ...n1.b },
      object: { ...n1.b },
      ...recordPred[1],
      ...source0,
      ...api0,
    };

    const record2_n1a_n3a = {
      subject: { ...n1.a },
      object: { ...n3.a },
      ...recordPred[2],
      ...source0,
      ...api0,
    };

    const record2_n3a_n1a = {
      ...record2_n1a_n3a,
      subject: { ...n3.a },
      object: { ...n1.a },
    };

    const record2_n1b_n3a = {
      subject: { ...n1.b },
      object: { ...n3.a },
      ...recordPred[2],
      ...source0,
      ...api0,
    };

    const record3_n1a_n4a = {
      subject: { ...n1.a },
      object: { ...n4.a },
      ...recordPred[3],
      ...source0,
      ...api0,
    };

    const record3_n1b_n4a = {
      subject: { ...n1.b },
      object: { ...n4.a },
      ...recordPred[3],
      ...source0,
      ...api0,
    };

    const record4_n2a_n5a = {
      subject: { ...n2.a },
      object: { ...n5.a },
      ...recordPred[4],
      ...source0,
      ...api0,
    };

    const record5_n3a_n5a = {
      subject: { ...n3.a },
      object: { ...n5.a },
      ...recordPred[5],
      ...source0,
      ...api0,
    };

    const record6_n4a_n5a = {
      subject: { ...n4.a },
      object: { ...n5.a },
      ...recordPred[6],
      ...source0,
      ...api0,
    };

    const record0_n0a_n1a_right_is_set = {
      ...record0_n0a_n1a,
      object: {
        ...set(n1.a),
      },
    };

    const record0_n0b_n1a_right_is_set = {
      ...record0_n0b_n1a,
      object: {
        ...set(n1.a),
      },
    };

    const record0_n0a_n1a_left_is_set = {
      ...record0_n0a_n1a,
      subject: {
        ...set(n0.a),
      },
    };

    const record0_n0b_n1a_left_is_set = {
      ...record0_n0b_n1a,
      subject: {
        ...set(n0.b),
      },
    };

    const record1_n1a_n2a_left_is_set = {
      ...record1_n1a_n2a,
      subject: {
        ...set(n1.a),
      },
    };

    const record1_n1a_n2b_left_is_set = {
      ...record1_n1a_n2b,
      subject: {
        ...set(n1.a),
      },
    };

    const record1_n2a_n1a_left_is_set = {
      ...record1_n2a_n1a,
      subject: {
        ...set(n2.a),
      },
    };

    const record1_n2b_n1a_left_is_set = {
      ...record1_n2b_n1a,
      subject: {
        ...set(n2.b),
      },
    };

    const record0_n0a_n1a_both_is_set = {
      ...record0_n0a_n1a,
      subject: {
        ...set(n0.a),
      },
      object: {
        ...set(n1.a),
      },
    };

    const record0_n0b_n1a_both_is_set = {
      ...record0_n0b_n1a,
      subject: {
        ...set(n0.b),
      },
      object: {
        ...set(n1.a),
      },
    };

    const record1_n2a_n1a_both_is_set = {
      ...record1_n2a_n1a,
      subject: {
        ...set(n2.a),
      },
      object: {
        ...set(n1.a),
      },
    };

    const record1_n2b_n1a_both_is_set = {
      ...record1_n2b_n1a,
      subject: {
        ...set(n2.b),
      },
      object: {
        ...set(n1.a),
      },
    };

    // start of synthetic record tests

    describe('repeat calls', () => {
      test('should get 0 results for update (0) & getResults (1)', async () => {
        const queryResultInner = new QueryResult({ provenanceUsesServiceProvider: false });
        const resultsInner = queryResultInner.getResults();
        expect(JSON.stringify(resultsInner)).toEqual(JSON.stringify([]));
      });

      // inputs all the same below here

      const queryResultOuter = new QueryResult({ provenanceUsesServiceProvider: false });
      let resultsOuter;
      test('just wrapping for async', async () => {
        await queryResultOuter.update({
          e0: {
            connected_to: ['e1'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1: {
            connected_to: ['e0'],
            records: [new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        resultsOuter = queryResultOuter.getResults();
      });

      test('should get same results: update (1) & getResults (1) vs. update (2) & getResults (1)', async () => {
        const queryResultInner = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResultInner.update({
          e0: {
            connected_to: ['e1'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1: {
            connected_to: ['e0'],
            records: [new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        await queryResultInner.update({
          e0: {
            connected_to: ['e1'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1: {
            connected_to: ['e0'],
            records: [new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        const resultsInner = queryResultInner.getResults();
        expect(JSON.stringify(resultsOuter)).toEqual(JSON.stringify(resultsInner));
      });

      test('should get same results: update (1) & getResults (1) vs. update (2) & getResults (2)', async () => {
        const queryResultInner = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResultInner.update({
          e0: {
            connected_to: ['e1'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1: {
            connected_to: ['e0'],
            records: [new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        await queryResultInner.update({
          e0: {
            connected_to: ['e1'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1: {
            connected_to: ['e0'],
            records: [new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        queryResultInner.getResults();
        const resultsInner = queryResultInner.getResults();
        expect(JSON.stringify(resultsOuter)).toEqual(JSON.stringify(resultsInner));
      });

      test('should get same results: update (1) & getResults (1) vs. update (1) & getResults (2)', async () => {
        const queryResultInner = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResultInner.update({
          e0: {
            connected_to: ['e1'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1: {
            connected_to: ['e0'],
            records: [new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        queryResultInner.getResults();
        const resultsInner = queryResultInner.getResults();
        expect(JSON.stringify(resultsOuter)).toEqual(JSON.stringify(resultsInner));
      });
    });

    describe('query graph: →', () => {
      test('should get 1 result with record: →', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: [],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0']);
        expect(results[0].analyses[0]).toHaveProperty('score');
      });

      test('should get 4 results for 4 different records per edge: 𝍬', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: [],
            records: [
              new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0a_n1b, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0b_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0b_n1b, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(4);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0']);
        expect(results[0].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[1].node_bindings).sort()).toEqual(['n0', 'n1']);
        expect(Object.keys(results[1].analyses[0].edge_bindings).sort()).toEqual(['e0']);
        expect(results[1].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[2].node_bindings).sort()).toEqual(['n0', 'n1']);
        expect(Object.keys(results[2].analyses[0].edge_bindings).sort()).toEqual(['e0']);
        expect(results[2].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[3].node_bindings).sort()).toEqual(['n0', 'n1']);
        expect(Object.keys(results[3].analyses[0].edge_bindings).sort()).toEqual(['e0']);
        expect(results[3].analyses[0]).toHaveProperty('score');
      });

      // TODO: Do we want to test for removing duplicates?
      test('should get 1 result for the same record repeated 4 times: 𝍬', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: [],
            records: [
              new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0']);
        expect(results[0].analyses[0]).toHaveProperty('score');
      });

      //      // TODO: this test fails. Do we need to handle this case?
      //      test('should get 1 result for the same record repeated twice and reversed twice: 𝍬', async () => {
      //        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
      //        await queryResult.update({
      //          "e1": {
      //            "connected_to": [],
      //            "records": [new Record(record1_n1a_n2a, config), new Record(record1_n1a_n2a, config), new Record(record1_n2a_n1a, config), new Record(record1_n2a_n1a, config)]
      //          },
      //        });
      //        const results = queryResult.getResults();
      //
      //        expect(results.length).toEqual(1);
      //
      //        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
      //          'n1', 'n2'
      //        ]);
      //        expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
      //          'e1'
      //        ]);
      //        expect(results[0]).toHaveProperty('score');
      //      });
      //
      //      // TODO: this one fails. Do we need to worry about this case?
      //      test('should get 2 results for the same record repeated twice and reversed twice: ⇉⇇', async () => {
      //        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
      //        await queryResult.update({
      //          "e1": {
      //            "connected_to": ["e1_reversed"],
      //            "records": [new Record(record1_n1a_n2a, config), new Record(record1_n1a_n2a, config)]
      //          },
      //          "e1_reversed": {
      //            "connected_to": ["e1"],
      //            "records": [new Record(record1_n2a_n1a, config), new Record(record1_n2a_n1a, config)]
      //          }
      //        });
      //        const results = queryResult.getResults();
      //
      //        expect(results.length).toEqual(2);
      //
      //        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
      //          'n1', 'n2'
      //        ]);
      //        expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
      //          'e1'
      //        ]);
      //        expect(results[0]).toHaveProperty('score');
      //
      //        expect(Object.keys(results[1].node_bindings).sort()).toEqual([
      //          'n2', 'n1'
      //        ]);
      //        expect(Object.keys(results[1].edge_bindings).sort()).toEqual([
      //          'e1_reversed'
      //        ]);
      //        expect(results[1]).toHaveProperty('score');
      //      });

      test('should get 1 result with 2 edge mappings when predicates differ: ⇉', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: [],
            records: [
              new Record(record0_n0a_n1a_pred0_api1, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0a_n1a_pred1_api1, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0']);

        expect(results[0].analyses[0].edge_bindings['e0'].length).toEqual(2);

        expect(results[0].analyses[0]).toHaveProperty('score');
      });

      // These two tests won't work until the KG edge ID assignment system is updated,
      // b/c we need it to take into account the API source.
      /*
      test('should get 1 result with 2 edge mappings when API sources differ: ⇉', async () => {
        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
        await queryResult.update({
          "e0": {
            "connected_to": [],
            "records": [new Record(record0_n0a_n1a_pred1_api0, config), new Record(record0_n0a_n1a_pred1_api1, config)]
          }
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
          'n0', 'n1'
        ]);
        expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
          'e0'
        ]);

        expect(results[0].edge_bindings['e0'].length).toEqual(2);

        expect(results[0]).toHaveProperty('score');
      });

      test('should get 1 result with 4 edge mappings when predicates & API sources differ: 𝍬', async () => {
        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
        await queryResult.update({
          "e0": {
            "connected_to": [],
            "records": [
              record0_n0a_n1a,
              record0_n0a_n1a_pred0_api1,
              record0_n0a_n1a_pred1_api0,
              record0_n0a_n1a_pred1_api1
            ]
          }
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
          'n0', 'n1'
        ]);
        expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
          'e0'
        ]);

        expect(results[0].edge_bindings['e0'].length).toEqual(4);

        expect(results[0]).toHaveProperty('score');
      });
      //*/
    });

    describe('query graph: →→', () => {
      test('should get 1 result with records: →→', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: ['e1'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1: {
            connected_to: ['e0'],
            records: [new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[0].analyses[0]).toHaveProperty('score');
      });
      test('should get 2 results with records: >-', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: ['e1'],
            records: [
              new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0b_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
          e1: {
            connected_to: ['e0'],
            records: [new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(2);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[0].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[1].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[1].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[1].analyses[0]).toHaveProperty('score');
      });
      test('should get 4 results with records: ><', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: ['e1'],
            records: [
              new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0b_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
          e1: {
            connected_to: ['e0'],
            records: [
              new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record1_n1a_n2b, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(4);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[0].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[1].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[1].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[1].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[2].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[2].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[2].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[3].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[3].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[3].analyses[0]).toHaveProperty('score');
      });
      test('should get 2 results with records: >< (is_set for n0)', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0_left_is_set: {
            connected_to: ['e1'],
            records: [
              new Record(record0_n0a_n1a_left_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0b_n1a_left_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
          e1: {
            connected_to: ['e0_left_is_set'],
            records: [
              new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record1_n1a_n2b, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(2);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0_with_is_set', 'n1', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0_left_is_set', 'e1']);
        expect(results[0].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[1].node_bindings).sort()).toEqual(['n0_with_is_set', 'n1', 'n2']);
        expect(Object.keys(results[1].analyses[0].edge_bindings).sort()).toEqual(['e0_left_is_set', 'e1']);
        expect(results[1].analyses[0]).toHaveProperty('score');
      });
      test('should get 4 results with records: >< (is_set for n1)', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0_right_is_set: {
            connected_to: ['e1_left_is_set'],
            records: [
              new Record(record0_n0a_n1a_right_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0b_n1a_right_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
          e1_left_is_set: {
            connected_to: ['e0_right_is_set'],
            records: [
              new Record(record1_n1a_n2a_left_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record1_n1a_n2b_left_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(4);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1_with_is_set', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0_right_is_set', 'e1_left_is_set']);
        expect(results[0].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[1].node_bindings).sort()).toEqual(['n0', 'n1_with_is_set', 'n2']);
        expect(Object.keys(results[1].analyses[0].edge_bindings).sort()).toEqual(['e0_right_is_set', 'e1_left_is_set']);
        expect(results[1].analyses[0]).toHaveProperty('score');
      });

      test('should get 1 result with records: >< (is_set for n0 and n2)', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0_left_is_set: {
            connected_to: ['e1_reversed_left_is_set'],
            records: [
              new Record(record0_n0a_n1a_left_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0b_n1a_left_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
          e1_reversed_left_is_set: {
            connected_to: ['e0_left_is_set'],
            records: [
              new Record(record1_n2a_n1a_left_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record1_n2b_n1a_left_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0_with_is_set', 'n1', 'n2_with_is_set']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual([
          'e0_left_is_set',
          'e1_reversed_left_is_set',
        ]);
        expect(results[0].analyses[0]).toHaveProperty('score');
      });

      test('should get 1 result with records: >< (is_set for n0, n1 and n2)', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0_both_is_set: {
            connected_to: ['e1_both_is_set'],
            records: [
              new Record(record0_n0a_n1a_both_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0b_n1a_both_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
          e1_both_is_set: {
            connected_to: ['e0_both_is_set'],
            records: [
              new Record(record1_n2a_n1a_both_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record1_n2b_n1a_both_is_set, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
          'n0_with_is_set',
          'n1_with_is_set',
          'n2_with_is_set',
        ]);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0_both_is_set', 'e1_both_is_set']);
        expect(results[0].analyses[0]).toHaveProperty('score');
      });

      test('should get 2 results with records: ⇉⇉', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: ['e1'],
            records: [
              new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0a_n1b, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
          e1: {
            connected_to: ['e0'],
            records: [
              new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record1_n1b_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(2);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[0].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[1].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[1].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[1].analyses[0]).toHaveProperty('score');
      });

      // TODO: Do we want to test for removing duplicates?
      test('should get 1 result with records: ⇉⇉ (duplicates)', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: ['e1'],
            records: [
              new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
          e1: {
            connected_to: ['e0'],
            records: [
              new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[0].analyses[0]).toHaveProperty('score');
      });
      test('should get 2 results with records: -<', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: ['e1'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1: {
            connected_to: ['e0'],
            records: [
              new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
              new Record(record1_n1a_n2b, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH),
            ],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(2);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[0].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[1].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[1].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[1].analyses[0]).toHaveProperty('score');
      });
      test('should get 1 result with records: →← (directionality does not match query graph)', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: ['e1_reversed'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1_reversed: {
            connected_to: ['e0'],
            records: [new Record(record1_n2a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1_reversed']);
        expect(results[0].analyses[0]).toHaveProperty('score');
      });

      test('should get 5k results when e0 has 100 records (50 connected, 50 not), and e1 has 10k (5k connected, 5k not)', async () => {
        /**
         * This test is intended to assess performance when handling a larger number of records.
         *
         * n0 -e0-> n1 -e1-> n2
         *
         * e0: 50 connected records + 50 unconnected records = 100 records
         * e1: 5k connected records + 5k unconnected records = 10k records
         *
         * common primaryIDs for records
         * @ n0: 1
         * @ n1: 50
         * @ n2: 100
         */
        const e0Records: Record[] = [];
        const e1Records: Record[] = [];

        const n0Count = 1;
        const n1Count = 50;
        const n2Count = 100;

        // just to ensure this matches the test name
        expect(n0Count * n1Count * n2Count).toEqual(5000);

        // generate connected records
        range(0, n1Count).forEach((n1Index) => {
          e0Records.push(
            new Record({
              subject: {
                ...n0.a,
              },
              object: {
                ...n1.a,
                curie: `n1_${n1Index}`,
              },
              predicate: 'biolink:record0_pred0',
              ...source0,
              ...api0,
            }),
          );

          range(0, n2Count).forEach((n2Index) => {
            e1Records.push(
              new Record({
                subject: {
                  ...n1.a,
                  curie: `n1_${n1Index}`,
                },
                object: {
                  ...n2.a,
                  curie: `n2_${n2Index}`,
                },
                predicate: 'biolink:record1_pred0',
                ...source1,
                ...api1,
              }),
            );
          });
        });

        // generate unconnected records
        range(0, n1Count).forEach((n1Index) => {
          e0Records.push(
            new Record({
              subject: {
                ...n0.a,
              },
              object: {
                ...n1.a,
                curie: 'n1_unconnected_e0record_',
              },
              predicate: 'biolink:record0_pred0',
              ...source0,
              ...api0,
            }),
          );

          range(0, n2Count).forEach((n2Index) => {
            e1Records.push(
              new Record({
                subject: {
                  ...n1.a,
                  curie: `n1_unconnected_e1record_${n1Index}`,
                },
                object: {
                  ...n2.a,
                  curie: `n2_${n2Index}`,
                },
                predicate: 'biolink:record1_pred0',
                ...source1,
                ...api1,
              }),
            );
          });
        });

        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: ['e1'],
            records: e0Records,
          },
          e1: {
            connected_to: ['e0'],
            records: e1Records,
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(5000);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[0].analyses[0]).toHaveProperty('score');

        expect(Object.keys(results[1].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[1].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[1].analyses[0]).toHaveProperty('score');
      });

      test('should get 1 result when e0 has 1 record, and e1 has 50k + 1 (1 connected, 50k not)', async () => {
        /**
         * n0 -e0-> n1 -e1-> n2
         *
         * e0: 1 connected record = 1 record
         * e1: 1 connected record + 50k unconnected records = 50,001 records
         *
         * primaryIDs for records
         * @ n0: 1 (1 common)
         * @ n1: 2 (1 common)
         * @ n2: 50001 (50001 common)
         */
        const e0Records: Record[] = [];
        const e1Records: Record[] = [];

        // generate connected records
        e0Records.push(
          new Record({
            subject: {
              ...n0.a,
            },
            object: {
              ...n1.a,
              curie: 'n1_connected',
            },
            predicate: 'biolink:record0_pred0',
            ...source0,
            ...api0,
          }),
        );

        e1Records.push(
          new Record({
            subject: {
              ...n1.a,
              curie: 'n1_connected',
            },
            object: {
              ...n2.a,
              curie: 'n2_connected',
            },
            predicate: 'biolink:record1_pred0',
            ...source1,
            ...api1,
          }),
        );

        // generate unconnected records
        range(0, 50000).forEach((n2Index) => {
          e1Records.push(
            new Record({
              subject: {
                ...n1.a,
                curie: 'n1_unconnected',
              },
              object: {
                ...n2.a,
                curie: `n2_unconnected_${n2Index}`,
              },
              predicate: 'biolink:record1_pred0',
              ...source1,
              ...api1,
            }),
          );
        });

        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: ['e1'],
            records: e0Records,
          },
          e1: {
            connected_to: ['e0'],
            records: e1Records,
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[0].analyses[0]).toHaveProperty('score');
      });

      /*
      describe('test large numbers of records', () => {
        // This group of tests is commented out, b/c they're too slow for regular testing.
        // They are intended to test performance when handling a large number of records.
        // Enable them only as needed.

        test('should get 50k results when e0 has 500 records, and e1 has 50k', () => {
          // n0 -e0-> n1 -e1-> n2
          //
          // e0: 500 records
          // e1: 500 * 100 = 50k records
          //
          // common primaryIDs for records
          // @ n0: 1
          // @ n1: 500
          // @ n2: 100

          const e0Records = [];
          const e1Records = [];

          const n0Count = 1;
          const n1Count = 500;
          const n2Count = 100;

          // just to ensure this matches the test name
          expect(n0Count * n1Count * n2Count).toEqual(50000);

          range(0, n1Count).forEach(n1Index => {
            e0Records.push({
              $edge_metadata: {
                trapi_qEdge_obj: e0,
                predicate: 'biolink:record0_pred0',
                source: 'source0',
                api_name: 'api0',
              },
              // n0
              subject: {
                normalizedInfo: [
                  {
                    primaryID: 'n0a',
                  },
                ],
              },
              // n1
              object: {
                normalizedInfo: [
                  {
                    primaryID: 'n1_' + n1Index,
                  },
                ],
              },
            });

            range(0, n2Count).forEach(n2Index => {
              e1Records.push({
                $edge_metadata: {
                  trapi_qEdge_obj: e1,
                  predicate: 'biolink:record1_pred0',
                  source: 'source1',
                  api_name: 'api1',
                },
                // n1
                subject: {
                  normalizedInfo: [
                    {
                      primaryID: 'n1_' + n1Index,
                    },
                  ],
                },
                // n2
                object: {
                  normalizedInfo: [
                    {
                      primaryID: 'n2_' + n2Index,
                    },
                  ],
                },
              });
            });
          });

          const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
          await queryResult.update({
            "e0": {
              "connected_to": ["e1"],
              "records": e0Records
            },
            "e1": {
              "connected_to": ["e0"],
              "records": e1Records
            }
          });
          const results = queryResult.getResults();

          expect(results.length).toEqual(50000);

          expect(Object.keys(results[0].node_bindings).sort()).toEqual([
            'n0', 'n1', 'n2'
          ]);
          expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
            'e0', 'e1'
          ]);
          expect(results[0]).toHaveProperty('score');

          expect(Object.keys(results[1].node_bindings).sort()).toEqual([
            'n0', 'n1', 'n2'
          ]);
          expect(Object.keys(results[1].edge_bindings).sort()).toEqual([
            'e0', 'e1'
          ]);
          expect(results[1]).toHaveProperty('score');
        });

        test('should get 50k results when e0 has 1k records (500 connected, 500 not), and e1 has 100k (50k connected, 50k not)', () => {
          // n0 -e0-> n1 -e1-> n2
          //
          // e0: 500 connected records + 500 unconnected records = 1k records
          // e1: 50k connected records + 50k unconnected records = 100k records
          //
          // common primaryIDs for records
          // @ n0: 1
          // @ n1: 500
          // @ n2: 100

          const e0Records = [];
          const e1Records = [];

          const n0Count = 1;
          const n1Count = 500;
          const n2Count = 100;

          // just to ensure this matches the test name
          expect(n0Count * n1Count * n2Count).toEqual(50000);

          // generate connected records
          range(0, n1Count).forEach(n1Index => {
            e0Records.push({
              $edge_metadata: {
                trapi_qEdge_obj: e0,
                predicate: 'biolink:record0_pred0',
                source: 'source0',
                api_name: 'api0',
              },
              // n0
              subject: {
                normalizedInfo: [
                  {
                    primaryID: 'n0a',
                  },
                ],
              },
              // n1
              object: {
                normalizedInfo: [
                  {
                    primaryID: 'n1_' + n1Index,
                  },
                ],
              },
            });

            range(0, n2Count).forEach(n2Index => {
              e1Records.push({
                $edge_metadata: {
                  trapi_qEdge_obj: e1,
                  predicate: 'biolink:record1_pred0',
                  source: 'source1',
                  api_name: 'api1',
                },
                // n1
                subject: {
                  normalizedInfo: [
                    {
                      primaryID: 'n1_' + n1Index,
                    },
                  ],
                },
                // n2
                object: {
                  normalizedInfo: [
                    {
                      primaryID: 'n2_' + n2Index,
                    },
                  ],
                },
              });
            });
          });

          // generate unconnected records
          range(0, n1Count).forEach(n1Index => {
            e0Records.push({
              $edge_metadata: {
                trapi_qEdge_obj: e0,
                predicate: 'biolink:record0_pred0',
                source: 'source0',
                api_name: 'api0',
              },
              // n0
              subject: {
                normalizedInfo: [
                  {
                    primaryID: 'n0a',
                  },
                ],
              },
              // n1
              object: {
                normalizedInfo: [
                  {
                    primaryID: 'n1_unconnected_e0record_' + n1Index,
                  },
                ],
              },
            });

            range(0, n2Count).forEach(n2Index => {
              e1Records.push({
                $edge_metadata: {
                  trapi_qEdge_obj: e1,
                  predicate: 'biolink:record1_pred0',
                  source: 'source1',
                  api_name: 'api1',
                },
                // n1
                subject: {
                  normalizedInfo: [
                    {
                      primaryID: 'n1_unconnected_e1record_' + n1Index,
                    },
                  ],
                },
                // n2
                object: {
                  normalizedInfo: [
                    {
                      primaryID: 'n2_' + n2Index,
                    },
                  ],
                },
              });
            });
          });

          const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
          await queryResult.update({
            "e0": {
              "connected_to": ["e1"],
              "records": e0Records
            },
            "e1": {
              "connected_to": ["e0"],
              "records": e1Records
            }
          });
          const results = queryResult.getResults();

          expect(results.length).toEqual(50000);

          expect(Object.keys(results[0].node_bindings).sort()).toEqual([
            'n0', 'n1', 'n2'
          ]);
          expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
            'e0', 'e1'
          ]);
          expect(results[0]).toHaveProperty('score');

          expect(Object.keys(results[1].node_bindings).sort()).toEqual([
            'n0', 'n1', 'n2'
          ]);
          expect(Object.keys(results[1].edge_bindings).sort()).toEqual([
            'e0', 'e1'
          ]);
          expect(results[1]).toHaveProperty('score');
        });

        test('should get 100k results when e0 has 1k connected records, and e1 has 100k', () => {
          // n0 -e0-> n1 -e1-> n2
          //
          // e0: 1k connected records
          // e1: 100k connected records
          //
          // common primaryIDs for records
          // @ n0: 1
          // @ n1: 1000
          // @ n2: 100

          const e0Records = [];
          const e1Records = [];

          const n0Count = 1;
          const n1Count = 1000;
          const n2Count = 100;

          // just to ensure this matches the test name
          expect(n0Count * n1Count * n2Count).toEqual(100000);

          range(0, n1Count).forEach(n1Index => {
            e0Records.push({
              $edge_metadata: {
                trapi_qEdge_obj: e0,
                predicate: 'biolink:record0_pred0',
                source: 'source0',
                api_name: 'api0',
              },
              // n0
              subject: {
                normalizedInfo: [
                  {
                    primaryID: 'n0a',
                  },
                ],
              },
              // n1
              object: {
                normalizedInfo: [
                  {
                    primaryID: 'n1_' + n1Index,
                  },
                ],
              },
            });

            range(0, n2Count).forEach(n2Index => {
              e1Records.push({
                $edge_metadata: {
                  trapi_qEdge_obj: e1,
                  predicate: 'biolink:record1_pred0',
                  source: 'source1',
                  api_name: 'api1',
                },
                // n1
                subject: {
                  normalizedInfo: [
                    {
                      primaryID: 'n1_' + n1Index,
                    },
                  ],
                },
                // n2
                object: {
                  normalizedInfo: [
                    {
                      primaryID: 'n2_' + n2Index,
                    },
                  ],
                },
              });
            });
          });

          const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
          await queryResult.update({
            "e0": {
              "connected_to": ["e1"],
              "records": e0Records
            },
            "e1": {
              "connected_to": ["e0"],
              "records": e1Records
            }
          });
          const results = queryResult.getResults();

          expect(results.length).toEqual(100000);

          expect(Object.keys(results[0].node_bindings).sort()).toEqual([
            'n0', 'n1', 'n2'
          ]);
          expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
            'e0', 'e1'
          ]);
          expect(results[0]).toHaveProperty('score');

          expect(Object.keys(results[1].node_bindings).sort()).toEqual([
            'n0', 'n1', 'n2'
          ]);
          expect(Object.keys(results[1].edge_bindings).sort()).toEqual([
            'e0', 'e1'
          ]);
          expect(results[1]).toHaveProperty('score');
        });
      });
      //*/
    });

    describe('query graph: →←', () => {
      test('should get 1 result with records: →←', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: ['e1_reversed'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1_reversed: {
            connected_to: ['e0'],
            records: [new Record(record1_n2a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1_reversed']);
        expect(results[0].analyses[0]).toHaveProperty('score');
      });

      test('should get 1 result with records: →→ (directionality does not match query graph)', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e0: {
            connected_to: ['e1'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1: {
            connected_to: ['e0'],
            records: [new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1']);
        expect(results[0].analyses[0]).toHaveProperty('score');
      });
    });

    describe('query graph: ←→', () => {
      test('should get 1 result for 1 record per edge: ←→', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        await queryResult.update({
          e1_reversed: {
            connected_to: ['e4'],
            records: [new Record(record1_n2a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e4: {
            connected_to: ['e1_reversed'],
            records: [new Record(record4_n2a_n5a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });
        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n1', 'n2', 'n5']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e1_reversed', 'e4']);
        expect(results[0].analyses[0]).toHaveProperty('score');
      });

      test('should get 0 results due to unconnected record: ←̽→', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });

        await queryResult.update({
          e1_reversed: {
            connected_to: ['e4'],
            records: [new Record(record1_n2b_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e4: {
            connected_to: ['e1_reversed'],
            records: [new Record(record4_n2a_n5a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });

        const results = queryResult.getResults();

        expect(results.length).toEqual(0);
      });
    });

    describe('query graph: -<', () => {
      /*
       *               -e1-> n2
       *   n0 -e0-> n1
       *               -e2-> n3
       */
      test('should get 1 result for 1 record per edge: →⇉⮆', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });

        await queryResult.update({
          e0: {
            connected_to: ['e1', 'e2'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1: {
            connected_to: ['e0', 'e2'],
            records: [new Record(record1_n1a_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e2: {
            connected_to: ['e0', 'e1'],
            records: [new Record(record2_n1a_n3a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });

        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2', 'n3']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1', 'e2']);

        expect(results[0].analyses[0]).toHaveProperty('score');
      });

      //      // TODO: this test fails
      //      /*
      //       *               -e1-> n2
      //       *   n0 <-e0- n1
      //       *               -e2-> n3
      //       */
      //      test('should get 1 result for 1 record per edge: ←⇉⮆', async () => {
      //        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
      //
      //        await queryResult.update({
      //          "e0_reversed": {
      //            "connected_to": ["e1", "e2"],
      //            "records": [new Record(record0_n1a_n0a, config)]
      //          },
      //          "e1": {
      //            "connected_to": ["e0_reversed", "e2"],
      //            "records": [new Record(record1_n1a_n2a, config)]
      //          },
      //          "e2": {
      //            "connected_to": ["e0_reversed", "e1"],
      //            "records": [new Record(record2_n1a_n3a, config)]
      //          },
      //        });
      //
      //        const results = queryResult.getResults();
      //
      //        expect(results.length).toEqual(1);
      //
      //        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
      //          'n0', 'n1', 'n2', 'n3'
      //        ]);
      //        expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
      //          'e0_reversed', 'e1', 'e2'
      //        ]);
      //
      //        expect(results[0]).toHaveProperty('score');
      //      });

      /*
       *               <-e1- n2
       *   n0 -e0-> n1
       *               -e2-> n3
       */
      test('should get 1 result for 1 record per edge: →⇆⮆', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });

        await queryResult.update({
          e0: {
            connected_to: ['e1_reversed', 'e2'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1_reversed: {
            connected_to: ['e0', 'e2'],
            records: [new Record(record1_n2a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e2: {
            connected_to: ['e0', 'e1_reversed'],
            records: [new Record(record2_n1a_n3a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });

        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2', 'n3']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1_reversed', 'e2']);

        expect(results[0].analyses[0]).toHaveProperty('score');
      });

      /*
       *               <-e1- n2
       *   n0 -e0-> n1
       *               <-e2- n3
       */
      test('should get 1 result for 1 record per edge: →⇇⮆', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });

        await queryResult.update({
          e0: {
            connected_to: ['e1_reversed', 'e2_reversed'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1_reversed: {
            connected_to: ['e0', 'e2_reversed'],
            records: [new Record(record1_n2a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e2_reversed: {
            connected_to: ['e0', 'e1_reversed'],
            records: [new Record(record2_n3a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });

        const results = queryResult.getResults();

        expect(results.length).toEqual(1);

        expect(Object.keys(results[0].node_bindings).sort()).toEqual(['n0', 'n1', 'n2', 'n3']);
        expect(Object.keys(results[0].analyses[0].edge_bindings).sort()).toEqual(['e0', 'e1_reversed', 'e2_reversed']);

        expect(results[0].analyses[0]).toHaveProperty('score');
      });

      /*
       *               x--> n2
       *   n0 ---> n1
       *               ---> n2
       */
      test('should get 0 results due to unconnected record: -<̽', async () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });

        await queryResult.update({
          e0: {
            connected_to: ['e1', 'e2'],
            records: [new Record(record0_n0a_n1a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e1: {
            connected_to: ['e0', 'e2'],
            records: [new Record(record1_n1b_n2a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
          e2: {
            connected_to: ['e0', 'e1'],
            records: [new Record(record2_n1a_n3a, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH)],
          },
        });

        const results = queryResult.getResults();

        expect(results.length).toEqual(0);
      });
    });

    //    // TODO: the following tests fail. Leaving them disabled below,
    //    // because Andrew said we don't have to handle cycles for now.
    //    describe('query graph: -ᗕᗒ', () => {
    //      /*
    //       *               -e1-> n2 -e4->
    //       *   n0 -e0-> n1 -e2-> n3 -e5-> n5
    //       *               -e3-> n4 -e6->
    //       */
    //
    ////      test('should get 1 result for 1 record per edge', async () => {
    ////        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
    ////
    ////        await queryResult.update({
    ////          "e0": {
    ////            "connected_to": ["e1", "e2", "e3"],
    ////            "records": [new Record(record0_n0a_n1a, config)]
    ////          },
    ////          "e1": {
    ////            "connected_to": ["e0", "e2", "e3", "e4"],
    ////            "records": [new Record(record1_n1a_n2a, config)]
    ////          },
    ////          "e2": {
    ////            "connected_to": ["e0", "e1", "e3", "e5"],
    ////            "records": [new Record(record2_n1a_n3a, config)]
    ////          },
    ////          "e3": {
    ////            "connected_to": ["e0", "e1",  "e2", "e6"],
    ////            "records": [new Record(record3_n1a_n4a, config)]
    ////          },
    ////          "e4": {
    ////            "connected_to": ["e1", "e5", "e6"],
    ////            "records": [new Record(record4_n2a_n5a, config)]
    ////          },
    ////          "e5": {
    ////            "connected_to": ["e2", "e4", "e6"],
    ////            "records": [new Record(record5_n3a_n5a, config)]
    ////          },
    ////          "e6": {
    ////            "connected_to": ["e3", "e4", "e5"],
    ////            "records": [new Record(record6_n4a_n5a, config)]
    ////          }
    ////        });
    ////
    ////        const results = queryResult.getResults();
    ////
    ////        expect(results.length).toEqual(1);
    ////
    ////        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[0]).toHaveProperty('score');
    ////      });
    //
    ////      test('should get 2 results for 2 records per edge at n0', async () => {
    ////        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
    ////
    ////        await queryResult.update({
    ////          "e0": {
    ////            "connected_to": ["e1", "e2", "e3"],
    ////            "records": [new Record(record0_n0a_n1a, config), new Record(record0_n0b_n1a, config)]
    ////          },
    ////          "e1": {
    ////            "connected_to": ["e0", "e2", "e3", "e4"],
    ////            "records": [new Record(record1_n1a_n2a, config)]
    ////          },
    ////          "e2": {
    ////            "connected_to": ["e0", "e1", "e3", "e5"],
    ////            "records": [new Record(record2_n1a_n3a, config)]
    ////          },
    ////          "e3": {
    ////            "connected_to": ["e0", "e1",  "e2", "e6"],
    ////            "records": [new Record(record3_n1a_n4a, config)]
    ////          },
    ////          "e4": {
    ////            "connected_to": ["e1", "e5", "e6"],
    ////            "records": [new Record(record4_n2a_n5a, config)]
    ////          },
    ////          "e5": {
    ////            "connected_to": ["e2", "e4", "e6"],
    ////            "records": [new Record(record5_n3a_n5a, config)]
    ////          },
    ////          "e6": {
    ////            "connected_to": ["e3", "e4", "e5"],
    ////            "records": [new Record(record6_n4a_n5a, config)]
    ////          }
    ////        });
    ////
    ////        const results = queryResult.getResults();
    ////
    ////        expect(results.length).toEqual(2);
    ////
    ////        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[0]).toHaveProperty('score');
    ////
    ////        expect(Object.keys(results[1].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[1].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[1]).toHaveProperty('score');
    ////      });
    ////
    ////      test('should get 2 results for 2 records per edge at n1', async () => {
    ////        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
    ////
    ////        await queryResult.update({
    ////          "e0": {
    ////            "connected_to": ["e1", "e2", "e3"],
    ////            "records": [new Record(record0_n0a_n1a, config), new Record(record0_n0a_n1b, config)]
    ////          },
    ////          "e1": {
    ////            "connected_to": ["e0", "e2", "e3", "e4"],
    ////            "records": [new Record(record1_n1a_n2a, config), new Record(record1_n1b_n2a, config)]
    ////          },
    ////          "e2": {
    ////            "connected_to": ["e0", "e1", "e3", "e5"],
    ////            "records": [new Record(record2_n1a_n3a, config), new Record(record2_n1b_n3a, config)]
    ////          },
    ////          "e3": {
    ////            "connected_to": ["e0", "e1",  "e2", "e6"],
    ////            "records": [new Record(record3_n1a_n4a, config), new Record(record3_n1b_n4a, config)]
    ////          },
    ////          "e4": {
    ////            "connected_to": ["e1", "e5", "e6"],
    ////            "records": [new Record(record4_n2a_n5a, config)]
    ////          },
    ////          "e5": {
    ////            "connected_to": ["e2", "e4", "e6"],
    ////            "records": [new Record(record5_n3a_n5a, config)]
    ////          },
    ////          "e6": {
    ////            "connected_to": ["e3", "e4", "e5"],
    ////            "records": [new Record(record6_n4a_n5a, config)]
    ////          }
    ////        });
    ////
    ////        const results = queryResult.getResults();
    ////
    ////        expect(results.length).toEqual(2);
    ////
    ////        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[0]).toHaveProperty('score');
    ////
    ////        expect(Object.keys(results[1].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[1].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[1]).toHaveProperty('score');
    ////      });
    ////
    ////      /*
    ////       *                 -e1-> n2a -e4->
    ////       *   n0a -e0-> n1a -e2-> n3a -e5-> n5a
    ////       *                 -e3-> n4a -e6->
    ////       *
    ////       *
    ////       *                 -e1-> n2a -e4->
    ////       *   n0a -e0-> n1b -e2-> n3a -e5-> n5a
    ////       *                 -e3-> n4a -e6->
    ////       *
    ////       *
    ////       *                 -e1-> n2a -e4->
    ////       *   n0b -e0-> n1a -e2-> n3a -e5-> n5a
    ////       *                 -e3-> n4a -e6->
    ////       */
    ////      test('should get 3 results for n0a→n1a, n0a→n1b, n0b→n1a', async () => {
    ////        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
    ////
    ////        await queryResult.update({
    ////          "e0": {
    ////            "connected_to": ["e1", "e2", "e3"],
    ////            "records": [new Record(record0_n0a_n1a, config), new Record(record0_n0a_n1b, config), new Record(record0_n0b_n1a, config)]
    ////          },
    ////          "e1": {
    ////            "connected_to": ["e0", "e2", "e3", "e4"],
    ////            "records": [new Record(record1_n1a_n2a, config), new Record(record1_n1b_n2a, config)]
    ////          },
    ////          "e2": {
    ////            "connected_to": ["e0", "e1", "e3", "e5"],
    ////            "records": [new Record(record2_n1a_n3a, config), new Record(record2_n1b_n3a, config)]
    ////          },
    ////          "e3": {
    ////            "connected_to": ["e0", "e1",  "e2", "e6"],
    ////            "records": [new Record(record3_n1a_n4a, config), new Record(record3_n1b_n4a, config)]
    ////          },
    ////          "e4": {
    ////            "connected_to": ["e1", "e5", "e6"],
    ////            "records": [new Record(record4_n2a_n5a, config)]
    ////          },
    ////          "e5": {
    ////            "connected_to": ["e2", "e4", "e6"],
    ////            "records": [new Record(record5_n3a_n5a, config)]
    ////          },
    ////          "e6": {
    ////            "connected_to": ["e3", "e4", "e5"],
    ////            "records": [new Record(record6_n4a_n5a, config)]
    ////          }
    ////        });
    ////
    ////        const results = queryResult.getResults();
    ////
    ////        expect(results.length).toEqual(3);
    ////
    ////        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[0]).toHaveProperty('score');
    ////
    ////        expect(Object.keys(results[1].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[1].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[1]).toHaveProperty('score');
    ////
    ////        expect(Object.keys(results[2].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[2].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[2]).toHaveProperty('score');
    ////      });
    ////
    ////      /*
    ////       *                 -e1-> n2a -e4->
    ////       *   n0a -e0-> n1a -e2-> n3a -e5-> n5a
    ////       *                 -e3-> n4a -e6->
    ////       *
    ////       *
    ////       *                 -e1-> n2a -e4->
    ////       *   n0a -e0-> n1b -e2-> n3a -e5-> n5a
    ////       *                 -e3-> n4a -e6->
    ////       *
    ////       *
    ////       *                 -e1-> n2a -e4->
    ////       *   n0b -e0-> n1a -e2-> n3a -e5-> n5a
    ////       *                 -e3-> n4a -e6->
    ////       *
    ////       *
    ////       *                 -e1-> n2a -e4->
    ////       *   n0b -e0-> n1b -e2-> n3a -e5-> n5a
    ////       *                 -e3-> n4a -e6->
    ////       */
    ////      test('should get 4 results for n0a→n1a, n0a→n1b, n0b→n1a, n0b→n1b', async () => {
    ////        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
    ////
    ////        await queryResult.update({
    ////          "e0": {
    ////            "connected_to": ["e1", "e2", "e3"],
    ////            "records": [new Record(record0_n0a_n1a, config), new Record(record0_n0a_n1b, config), new Record(record0_n0b_n1a, config), new Record(record0_n0b_n1b, config)]
    ////          },
    ////          "e1": {
    ////            "connected_to": ["e0", "e2", "e3", "e4"],
    ////            "records": [new Record(record1_n1a_n2a, config), new Record(record1_n1b_n2a, config)]
    ////          },
    ////          "e2": {
    ////            "connected_to": ["e0", "e1", "e3", "e5"],
    ////            "records": [new Record(record2_n1a_n3a, config), new Record(record2_n1b_n3a, config)]
    ////          },
    ////          "e3": {
    ////            "connected_to": ["e0", "e1",  "e2", "e6"],
    ////            "records": [new Record(record3_n1a_n4a, config), new Record(record3_n1b_n4a, config)]
    ////          },
    ////          "e4": {
    ////            "connected_to": ["e1", "e5", "e6"],
    ////            "records": [new Record(record4_n2a_n5a, config)]
    ////          },
    ////          "e5": {
    ////            "connected_to": ["e2", "e4", "e6"],
    ////            "records": [new Record(record5_n3a_n5a, config)]
    ////          },
    ////          "e6": {
    ////            "connected_to": ["e3", "e4", "e5"],
    ////            "records": [new Record(record6_n4a_n5a, config)]
    ////          }
    ////        });
    ////
    ////        const results = queryResult.getResults();
    ////
    ////        expect(results.length).toEqual(4);
    ////
    ////        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[0]).toHaveProperty('score');
    ////
    ////        expect(Object.keys(results[1].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[1].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[1]).toHaveProperty('score');
    ////
    ////        expect(Object.keys(results[2].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[2].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[2]).toHaveProperty('score');
    ////
    ////        expect(Object.keys(results[3].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[3].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[3]).toHaveProperty('score');
    ////      });
    //
    //      test('should get 0 results due to unconnected record at n1 (n1a vs. n1b)', async () => {
    //        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
    //
    //        await queryResult.update({
    //          "e0": {
    //            "connected_to": ["e1", "e2", "e3"],
    //            "records": [new Record(record0_n0a_n1a, config)]
    //          },
    //          "e1": {
    //            "connected_to": ["e0", "e2", "e3", "e4"],
    //            "records": [new Record(record1_n1b_n2a, config)]
    //          },
    //          "e2": {
    //            "connected_to": ["e0", "e1", "e3", "e5"],
    //            "records": [new Record(record2_n1a_n3a, config)]
    //          },
    //          "e3": {
    //            "connected_to": ["e0", "e1",  "e2", "e6"],
    //            "records": [new Record(record3_n1a_n4a, config)]
    //          },
    //          "e4": {
    //            "connected_to": ["e1", "e5", "e6"],
    //            "records": [new Record(record4_n2a_n5a, config)]
    //          },
    //          "e5": {
    //            "connected_to": ["e2", "e4", "e6"],
    //            "records": [new Record(record5_n3a_n5a, config)]
    //          },
    //          "e6": {
    //            "connected_to": ["e3", "e4", "e5"],
    //            "records": [new Record(record6_n4a_n5a, config)]
    //          }
    //        });
    //
    //        const results = queryResult.getResults();
    //
    //        expect(results.length).toEqual(0);
    //      });
    //
    ////      test('should get 1 result & ignore unconnected record', async () => {
    ////        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
    ////
    ////        await queryResult.update({
    ////          "e0": {
    ////            "connected_to": ["e1", "e2", "e3"],
    ////            "records": [new Record(record0_n0a_n1a, config)]
    ////          },
    ////          "e1": {
    ////            "connected_to": ["e0", "e2", "e3", "e4"],
    ////            "records": [new Record(record1_n1a_n2a, config), new Record(record1_n1b_n2a, config)]
    ////          },
    ////          "e2": {
    ////            "connected_to": ["e0", "e1", "e3", "e5"],
    ////            "records": [new Record(record2_n1a_n3a, config)]
    ////          },
    ////          "e3": {
    ////            "connected_to": ["e0", "e1",  "e2", "e6"],
    ////            "records": [new Record(record3_n1a_n4a, config)]
    ////          },
    ////          "e4": {
    ////            "connected_to": ["e1", "e5", "e6"],
    ////            "records": [new Record(record4_n2a_n5a, config)]
    ////          },
    ////          "e5": {
    ////            "connected_to": ["e2", "e4", "e6"],
    ////            "records": [new Record(record5_n3a_n5a, config)]
    ////          },
    ////          "e6": {
    ////            "connected_to": ["e3", "e4", "e5"],
    ////            "records": [new Record(record6_n4a_n5a, config)]
    ////          }
    ////        });
    ////
    ////        const results = queryResult.getResults();
    ////
    ////        expect(results.length).toEqual(1);
    ////
    ////        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[0]).toHaveProperty('score');
    ////      });
    ////
    ////      test('should get 1 result & ignore 4 unconnected records', async () => {
    ////        const queryResult = new QueryResult(provenanceUsesServiceProvider = false);
    ////
    ////        await queryResult.update({
    ////          "e0": {
    ////            "connected_to": ["e1", "e2", "e3"],
    ////            "records": [new Record(record0_n0a_n1a, config), new Record(record0_n0a_n1b, config)]
    ////          },
    ////          "e1": {
    ////            "connected_to": ["e0", "e2", "e3", "e4"],
    ////            "records": [new Record(record1_n1a_n2a, config), new Record(record1_n1a_n2b, config), new Record(record1_n1b_n2a, config), new Record(record1_n1b_n2b, config)]
    ////          },
    ////          "e2": {
    ////            "connected_to": ["e0", "e1", "e3", "e5"],
    ////            "records": [new Record(record2_n1a_n3a, config)]
    ////          },
    ////          "e3": {
    ////            "connected_to": ["e0", "e1",  "e2", "e6"],
    ////            "records": [new Record(record3_n1a_n4a, config)]
    ////          },
    ////          "e4": {
    ////            "connected_to": ["e1", "e5", "e6"],
    ////            "records": [new Record(record4_n2a_n5a, config)]
    ////          },
    ////          "e5": {
    ////            "connected_to": ["e2", "e4", "e6"],
    ////            "records": [new Record(record5_n3a_n5a, config)]
    ////          },
    ////          "e6": {
    ////            "connected_to": ["e3", "e4", "e5"],
    ////            "records": [new Record(record6_n4a_n5a, config)]
    ////          }
    ////        });
    ////
    ////        const results = queryResult.getResults();
    ////
    ////        expect(results.length).toEqual(1);
    ////
    ////        expect(Object.keys(results[0].node_bindings).sort()).toEqual([
    ////          'n0', 'n1', 'n2', 'n3', 'n4', 'n5'
    ////        ]);
    ////        expect(Object.keys(results[0].edge_bindings).sort()).toEqual([
    ////          'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ////        ]);
    ////        expect(results[0]).toHaveProperty('score');
    ////      });
    //    });
    test('Records with different predicates should not be merged', async () => {
      const record1 = new Record({
        subject: {
          original: 'NCBIGene:3265',
          qNodeID: 'n1',
          isSet: false,
          curie: 'NCBIGene:3265',
          UMLS: ['C0079471', 'C1569842'],
          semanticType: ['Gene'],
          label: 'HRAS',
          attributes: {},
        },
        object: {
          original: 'BIOPLANET:bioplanet_1498',
          qNodeID: 'n0',
          isSet: false,
          curie: 'BIOPLANET:bioplanet_1498',
          semanticType: ['Pathway'],
          label: 'BIOPLANET:bioplanet_1498',
          attributes: {},
        },
        predicate: 'biolink:actively_involved_in',
        mappedResponse: {
          pathway_name: 'Cell surface interactions at the vascular wall',
          pathway_categories: [
            'Cardiovascular disease',
            'Circulatory system',
            'Hemostasis',
            'Immune disease',
            'Immune system',
            'Infectious disease',
          ],
          gene_symbol: 'HRAS',
        },
      });
      const record2 = new Record({
        subject: {
          original: 'NCBIGene:3265',
          qNodeID: 'n1',
          isSet: false,
          curie: 'NCBIGene:3265',
          UMLS: ['C0079471', 'C1569842'],
          semanticType: ['Gene'],
          label: 'HRAS',
          attributes: {},
        },
        object: {
          original: 'BIOPLANET:bioplanet_1498',
          qNodeID: 'n0',
          isSet: false,
          curie: 'BIOPLANET:bioplanet_1498',
          semanticType: ['Pathway'],
          label: 'BIOPLANET:bioplanet_1498',
          attributes: {},
        },
        predicate: 'biolink:related_to',
        mappedResponse: {
          pathway_name: 'Cell surface interactions at the vascular wall',
          pathway_categories: [
            'Cardiovascular disease',
            'Circulatory system',
            'Hemostasis',
            'Immune disease',
            'Immune system',
            'Infectious disease',
          ],
          gene_symbol: 'HRAS',
        },
      });
      const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
      await queryResult.update({
        e01: {
          connected_to: [],
          records: [record1, record2],
        },
      });
      expect(queryResult.getResults()[0].analyses[0].edge_bindings.e01.length).toEqual(2);
    });
  });

  describe('Graph structure', () => {
    describe('Initial QNode ID selection', () => {
      const exampleRecordsByQEdgeID = {
        e0: {
          connected_to: ['e1', 'e2'],
          records: [
            new Record({
              //@ts-expect-error: partial data for specific test
              subject: {
                qNodeID: 'n2',
              },
              //@ts-expect-error: partial data for specific test
              object: {
                qNodeID: 'n1',
              },
            }),
          ],
        },
        e1: {
          connected_to: ['e0'],
          records: [
            new Record({
              //@ts-expect-error: partial data for specific test
              subject: {
                qNodeID: 'n2',
              },
              //@ts-expect-error: partial data for specific test
              object: {
                qNodeID: 'n3',
              },
            }),
          ],
        },
        e2: {
          connected_to: ['e0'],
          records: [
            new Record({
              //@ts-expect-error: partial data for specific test
              subject: {
                qNodeID: 'n0',
              },
              //@ts-expect-error: partial data for specific test
              object: {
                qNodeID: 'n1',
              },
            }),
          ],
        },
      };

      test('Should select leaf node', () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        const [[initialNode, initialEdge]] = queryResult._getValidInitialPairs(exampleRecordsByQEdgeID);
        expect(initialNode).toEqual('n3');
        expect(initialEdge).toEqual('e1');
      });
      test('Should select leaf node with fewest records on associated edge', () => {
        const queryResult = new QueryResult({ provenanceUsesServiceProvider: false });
        const example = cloneDeep(exampleRecordsByQEdgeID);
        //@ts-expect-error: specific test behavior
        example.e1.records.push({ fake: true });
        const [[initialNode, initialEdge]] = queryResult._getValidInitialPairs(example);
        expect(initialNode).toEqual('n0');
        expect(initialEdge).toEqual('e2');
      });
    });
  });
});
