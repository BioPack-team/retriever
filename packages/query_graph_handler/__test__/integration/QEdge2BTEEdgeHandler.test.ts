jest.mock('axios');
import axios from 'axios';

import QNode from '../../src/query_node';
import QEdge from '../../src/query_edge';
import NodeUpdateHandler from '../../src/update_nodes';

describe('Testing NodeUpdateHandler Module', () => {
  const gene_node1 = new QNode({ id: 'n1', categories: ['Gene'], ids: ['NCBIGene:1017'] });
  const node1_equivalent_ids = {
    'NCBIGene:1017': [
      {
        semanticTypes: [],
        db_ids: {
          NCBIGene: ['1017'],
          SYMBOL: ['CDK2'],
        },
      },
    ],
  };

  const gene_node2 = new QNode({ id: 'n2', categories: ['Gene'], ids: ['NCBIGene:1017', 'NCBIGene:1018'] });
  const gene_node1_with_id_annotated = new QNode({ id: 'n1', categories: ['Gene'], ids: ['NCBIGene:1017'] });
  //@ts-expect-error: partial data for specific test scope
  gene_node1_with_id_annotated.setEquivalentIDs(node1_equivalent_ids);
  //gene_node2.setEquivalentIDs(node2_equivalent_ids);
  const chemical_node1 = new QNode({ id: 'n3', categories: ['SmallMolecule'] });
  const edge1 = new QEdge({ id: 'e01', subject: gene_node1, object: chemical_node1 });
  const edge2 = new QEdge({ id: 'e02', subject: gene_node1_with_id_annotated, object: chemical_node1 });
  const edge3 = new QEdge({ id: 'e04', subject: gene_node2, object: chemical_node1 });
  const edge4 = new QEdge({ id: 'e05', object: gene_node2, subject: chemical_node1 });

  describe('Testing _getCuries function', () => {
    test('test edge with one curie input return an array of one', () => {
      (axios.post as jest.Mock).mockResolvedValueOnce({
        data: {
          'NCBIGene:1017': {
            id: { identifier: 'NCBIGene:1017', label: 'CDK2' },
            equivalent_identifiers: [
              { identifier: 'NCBIGene:1017', label: 'CDK2' },
              { identifier: 'ENSEMBL:ENSG00000123374' },
              { identifier: 'HGNC:1771', label: 'CDK2' },
              { identifier: 'OMIM:116953' },
              { identifier: 'UMLS:C1332733', label: 'CDK2 gene' },
              {
                identifier: 'UniProtKB:A0A024RB10',
                label: 'A0A024RB10_HUMAN Cyclin-dependent kinase 2, isoform CRA_a (trembl)',
              },
              {
                identifier: 'UniProtKB:A0A024RB77',
                label: 'A0A024RB77_HUMAN Cyclin-dependent kinase 2, isoform CRA_b (trembl)',
              },
              {
                identifier: 'UniProtKB:B4DDL9',
                label:
                  'B4DDL9_HUMAN cDNA FLJ54979, highly similar to Homo sapiens cyclin-dependent kinase 2 (CDK2), transcript variant 2, mRNA (trembl)',
              },
              { identifier: 'UniProtKB:E7ESI2', label: 'E7ESI2_HUMAN Cyclin-dependent kinase 2 (trembl)' },
              { identifier: 'ENSEMBL:ENSP00000393605' },
              { identifier: 'UniProtKB:G3V5T9', label: 'G3V5T9_HUMAN Cyclin-dependent kinase 2 (trembl)' },
              { identifier: 'ENSEMBL:ENSP00000452514' },
              { identifier: 'UniProtKB:P24941', label: 'CDK2_HUMAN Cyclin-dependent kinase 2 (sprot)' },
              { identifier: 'PR:P24941', label: 'cyclin-dependent kinase 2 (human)' },
              { identifier: 'UMLS:C0108855', label: 'CDK2 protein, human' },
            ],
            type: [
              'biolink:Gene',
              'biolink:GeneOrGeneProduct',
              'biolink:GenomicEntity',
              'biolink:ChemicalEntityOrGeneOrGeneProduct',
              'biolink:PhysicalEssence',
              'biolink:OntologyClass',
              'biolink:BiologicalEntity',
              'biolink:NamedThing',
              'biolink:Entity',
              'biolink:PhysicalEssenceOrOccurrent',
              'biolink:ThingWithTaxon',
              'biolink:MacromolecularMachineMixin',
              'biolink:Protein',
              'biolink:GeneProductMixin',
              'biolink:Polypeptide',
              'biolink:ChemicalEntityOrProteinOrPolypeptide',
            ],
            information_content: 100,
          },
        },
      });
      const nodeUpdater = new NodeUpdateHandler([edge1]);
      const res = nodeUpdater._getCuries([edge1]);
      expect(res).toHaveProperty('Gene', ['NCBIGene:1017']);
    });

    test('test edge with multiple curie input return an array with multiple items', () => {
      (axios.post as jest.Mock).mockResolvedValueOnce({
        data: {
          'NCBIGene:1017': {
            id: { identifier: 'NCBIGene:1017', label: 'CDK2' },
            equivalent_identifiers: [
              { identifier: 'NCBIGene:1017', label: 'CDK2' },
              { identifier: 'ENSEMBL:ENSG00000123374' },
              { identifier: 'HGNC:1771', label: 'CDK2' },
              { identifier: 'OMIM:116953' },
              { identifier: 'UMLS:C1332733', label: 'CDK2 gene' },
              {
                identifier: 'UniProtKB:A0A024RB10',
                label: 'A0A024RB10_HUMAN Cyclin-dependent kinase 2, isoform CRA_a (trembl)',
              },
              {
                identifier: 'UniProtKB:A0A024RB77',
                label: 'A0A024RB77_HUMAN Cyclin-dependent kinase 2, isoform CRA_b (trembl)',
              },
              {
                identifier: 'UniProtKB:B4DDL9',
                label:
                  'B4DDL9_HUMAN cDNA FLJ54979, highly similar to Homo sapiens cyclin-dependent kinase 2 (CDK2), transcript variant 2, mRNA (trembl)',
              },
              { identifier: 'UniProtKB:E7ESI2', label: 'E7ESI2_HUMAN Cyclin-dependent kinase 2 (trembl)' },
              { identifier: 'ENSEMBL:ENSP00000393605' },
              { identifier: 'UniProtKB:G3V5T9', label: 'G3V5T9_HUMAN Cyclin-dependent kinase 2 (trembl)' },
              { identifier: 'ENSEMBL:ENSP00000452514' },
              { identifier: 'UniProtKB:P24941', label: 'CDK2_HUMAN Cyclin-dependent kinase 2 (sprot)' },
              { identifier: 'PR:P24941', label: 'cyclin-dependent kinase 2 (human)' },
              { identifier: 'UMLS:C0108855', label: 'CDK2 protein, human' },
            ],
            type: [
              'biolink:Gene',
              'biolink:GeneOrGeneProduct',
              'biolink:GenomicEntity',
              'biolink:ChemicalEntityOrGeneOrGeneProduct',
              'biolink:PhysicalEssence',
              'biolink:OntologyClass',
              'biolink:BiologicalEntity',
              'biolink:NamedThing',
              'biolink:Entity',
              'biolink:PhysicalEssenceOrOccurrent',
              'biolink:ThingWithTaxon',
              'biolink:MacromolecularMachineMixin',
              'biolink:Protein',
              'biolink:GeneProductMixin',
              'biolink:Polypeptide',
              'biolink:ChemicalEntityOrProteinOrPolypeptide',
            ],
            information_content: 100,
          },
          'NCBIGene:1018': {
            id: { identifier: 'NCBIGene:1018', label: 'CDK3' },
            equivalent_identifiers: [
              { identifier: 'NCBIGene:1018', label: 'CDK3' },
              { identifier: 'ENSEMBL:ENSG00000250506' },
              { identifier: 'HGNC:1772', label: 'CDK3' },
              { identifier: 'OMIM:123828' },
              { identifier: 'UMLS:C1332734', label: 'CDK3 gene' },
              { identifier: 'UniProtKB:Q00526', label: 'CDK3_HUMAN Cyclin-dependent kinase 3 (sprot)' },
              { identifier: 'PR:Q00526', label: 'cyclin-dependent kinase 3 (human)' },
              { identifier: 'ENSEMBL:ENSP00000400088' },
              { identifier: 'ENSEMBL:ENSP00000410561' },
              { identifier: 'UMLS:C1447440', label: 'CDK3 protein, human' },
            ],
            type: [
              'biolink:Gene',
              'biolink:GeneOrGeneProduct',
              'biolink:GenomicEntity',
              'biolink:ChemicalEntityOrGeneOrGeneProduct',
              'biolink:PhysicalEssence',
              'biolink:OntologyClass',
              'biolink:BiologicalEntity',
              'biolink:NamedThing',
              'biolink:Entity',
              'biolink:PhysicalEssenceOrOccurrent',
              'biolink:ThingWithTaxon',
              'biolink:MacromolecularMachineMixin',
              'biolink:Protein',
              'biolink:GeneProductMixin',
              'biolink:Polypeptide',
              'biolink:ChemicalEntityOrProteinOrPolypeptide',
            ],
            information_content: 100,
          },
          'PUBCHEM:5070': null,
        },
      });
      const nodeUpdater = new NodeUpdateHandler([edge3]);
      const res = nodeUpdater._getCuries([edge3]);
      expect(res.Gene.length).toEqual(2);
    });

    // test deprecated: proper update handling outside of updater ensures minimal redundancy
    // test('test edge with input node annotated should return an empty array', () => {
    //   const nodeUpdater = new NodeUpdateHandler([edge2]);
    //   const res = nodeUpdater._getCuries([edge2]);
    //   expect(res).toEqual({});
    // });

    test('test edge with input on object end should be handled', () => {
      const nodeUpdater = new NodeUpdateHandler([edge4]);
      const res = nodeUpdater._getCuries([edge4]);
      expect(res.Gene.length).toEqual(2);
    });
  });

  describe('Testing _getEquivalentIDs function', () => {
    test('test edge with one curie input return an object with one key', async () => {
      const nodeUpdater = new NodeUpdateHandler([edge1]);
      const res = await nodeUpdater._getEquivalentIDs({ Gene: ['NCBIGene:1017'] });
      expect(res).toHaveProperty('NCBIGene:1017');
    });

    test('test edge with multiple curie input return an object with multiple key', async () => {
      const nodeUpdater = new NodeUpdateHandler([edge1]);
      const res = await nodeUpdater._getEquivalentIDs({
        Gene: ['NCBIGene:1017', 'NCBIGene:1018'],
        SmallMolecule: ['PUBCHEM:5070'],
      });
      expect(res).toHaveProperty('NCBIGene:1017');
      expect(res).toHaveProperty('NCBIGene:1018');
      expect(res).toHaveProperty('PUBCHEM:5070');
    });
  });
});
