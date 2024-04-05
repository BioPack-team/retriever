import DefaultIDResolver from '../../src/resolve/default_resolver';
import { ResolvableBioEntity } from '../../src/bioentity/valid_bioentity';
import { IrresolvableBioEntity } from '../../src/bioentity/irresolvable_bioentity';

jest.setTimeout(30000)
describe.skip("Test ID Resolver", () => {
    test("Test valid inputs should be corretly resolved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Gene": ["NCBIGene:1017"] });
        expect(res).toHaveProperty("NCBIGene:1017");
        expect(res['NCBIGene:1017']).toHaveLength(1);
        expect(res['NCBIGene:1017'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['NCBIGene:1017'][0].primaryID).toEqual("NCBIGene:1017");
        expect(res['NCBIGene:1017'][0].label).toEqual("CDK2")
    })

    test("Test symbol should be resolved to corresponding human gene", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Gene": ["SYMBOL:VAMP2"] });
        expect(res).toHaveProperty("SYMBOL:VAMP2");
        expect(res['SYMBOL:VAMP2']).toHaveLength(1);
        expect(res['SYMBOL:VAMP2'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['SYMBOL:VAMP2'][0].primaryID).toEqual("NCBIGene:6844");
        expect(res['SYMBOL:VAMP2'][0].label).toEqual("VAMP2")
    })

    test("Test LINCS ID should be resolved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "SmallMolecule": ["LINCS:LSM-2471"] });
        expect(res).toHaveProperty("LINCS:LSM-2471");
        expect(res['LINCS:LSM-2471']).toHaveLength(1);
        expect(res['LINCS:LSM-2471'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['LINCS:LSM-2471'][0].primaryID).toEqual("PUBCHEM.COMPOUND:5070");
        expect(res['LINCS:LSM-2471'][0].dbIDs.LINCS).toEqual(["LSM-2471"]);
    })

    test("Test Protein Uniprot ID should be resolved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Protein": ["UniProtKB:P24941"] });
        expect(res).toHaveProperty("UniProtKB:P24941");
        expect(res['UniProtKB:P24941']).toHaveLength(1);
        expect(res['UniProtKB:P24941'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['UniProtKB:P24941'][0].primaryID).toEqual("UniProtKB:P24941");
        expect(res['UniProtKB:P24941'][0].dbIDs.ENSEMBL).toContain("ENSP00000243067")
    })

    test("Test Protein ENSEMBL ID should be resolved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Protein": ["ENSEMBL:ENSP00000243067"] });
        expect(res).toHaveProperty("ENSEMBL:ENSP00000243067");
        expect(res['ENSEMBL:ENSP00000243067']).toHaveLength(1);
        expect(res['ENSEMBL:ENSP00000243067'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['ENSEMBL:ENSP00000243067'][0].primaryID).toEqual("UniProtKB:P24941");
        expect(res['ENSEMBL:ENSP00000243067'][0].dbIDs.ENSEMBL).toContain("ENSP00000243067")
    })

    test("records from a query with multiple hits should all be collected", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Disease": ["OMIM:307030"] });
        expect(res).toHaveProperty("OMIM:307030");
        expect(res['OMIM:307030']).toHaveLength(1);
        expect(res['OMIM:307030'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['OMIM:307030'][0].primaryID).toEqual("MONDO:0010613");
        expect(res['OMIM:307030'][0].dbIDs.UMLS).toContain("C0268418");
        expect(res['OMIM:307030'][0].dbIDs.UMLS).toContain("C0574108");
    })

    test("Test valid inputs should be corretly resolved using Disease GARD ID", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Disease": ["GARD:4206"] });
        expect(res).toHaveProperty("GARD:4206");
        expect(res['GARD:4206']).toHaveLength(1);
        expect(res['GARD:4206'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['GARD:4206'][0].primaryID).toEqual("MONDO:0015278");
    })

    test("Test BioThings output include integer should be converted to string", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "SmallMolecule": ["CHEMBL.COMPOUND:CHEMBL744"] });
        expect(res['CHEMBL.COMPOUND:CHEMBL744'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['CHEMBL.COMPOUND:CHEMBL744'][0].dbIDs["PUBCHEM.COMPOUND"]).toEqual(["5070"]);
    })

    test("Test valid inputs from multiple semantic types should be corretly resolved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Gene": ["NCBIGene:1017"], "SmallMolecule": ["DRUGBANK:DB01609"] });
        expect(res).toHaveProperty("NCBIGene:1017");
        expect(res['NCBIGene:1017']).toHaveLength(1);
        expect(res['NCBIGene:1017'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['NCBIGene:1017'][0].primaryID).toEqual("NCBIGene:1017");
        expect(res['NCBIGene:1017'][0].label).toEqual("CDK2");
        expect(res).toHaveProperty("DRUGBANK:DB01609");
        expect(res["DRUGBANK:DB01609"]).toHaveLength(1);
        expect(res['DRUGBANK:DB01609'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['DRUGBANK:DB01609'][0].label.toUpperCase()).toEqual("DEFERASIROX");
    })

    test("Test Irresolvable inputs should be part of the result", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Gene": ["NCBIGene:1017", "kkk:123"] });
        expect(res).toHaveProperty("kkk:123");
        expect(res['kkk:123']).toHaveLength(1);
        expect(res['kkk:123'][0]).toBeInstanceOf(IrresolvableBioEntity);
        expect(res['kkk:123'][0].primaryID).toEqual('kkk:123');
        expect(res['kkk:123'][0].label).toEqual('kkk:123')
    })
    
    // uses old ID resolver and has issues right now...
    test.skip("test large batch of inputs should be correctly resolved", async () => {
        const fakeNCBIGeneInputs = [...Array(1990).keys()].map(item => 'NCBIGene:' + item.toString());
        const fakeOMIMGeneInputs = [...Array(2300).keys()].map(item => "OMIM:" + item.toString());
        const fakeDrugbankInputs = [...Array(3500).keys()].map(item => "DRUGBANK:DB00" + item.toString());
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({
            Gene: [...fakeNCBIGeneInputs, ...fakeOMIMGeneInputs],
            SmallMolecule: fakeDrugbankInputs
        })
        expect(Object.keys(res)).toHaveLength(fakeDrugbankInputs.length + fakeNCBIGeneInputs.length + fakeOMIMGeneInputs.length);
        expect(res['OMIM:0'][0]).toBeInstanceOf(IrresolvableBioEntity)

    })

    test("Test inputs with undefined semanticType should be corretly resolved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "undefined": ["NCBIGene:1017"] });
        expect(res).toHaveProperty("NCBIGene:1017");
        expect(res['NCBIGene:1017'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['NCBIGene:1017'][0].primaryID).toEqual("NCBIGene:1017");
        expect(res['NCBIGene:1017'][0].label).toEqual("CDK2");
        expect(res['NCBIGene:1017'][0].semanticType).toEqual("Gene");
    })

    test("Test inputs with undefined semanticType and could be mapped to multiple semantictypes should be corretly resolved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "undefined": ["UMLS:C0008780"] });
        expect(res).toHaveProperty("UMLS:C0008780");
        const valid = res["UMLS:C0008780"].filter(rec => rec instanceof ResolvableBioEntity);
        expect(valid).toHaveLength(2);
        expect(valid[0].semanticType).toEqual("PhenotypicFeature");
        expect(valid[1].primaryID).toEqual("MONDO:0016575");
        expect(valid[1].label).toEqual("primary ciliary dyskinesia");
        expect(valid[1].semanticType).toEqual("Disease");
    })

    test("Test inputs with undefined semanticType and could be mapped to multiple semantictypes using OMIM ID as example should be corretly resolved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "undefined": ["OMIM:116953"] });
        expect(res).toHaveProperty("OMIM:116953");
        expect(res['OMIM:116953'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['OMIM:116953'][0].primaryID).toEqual("NCBIGene:1017");
        expect(res['OMIM:116953'][0].label).toEqual("CDK2");
        expect(res['OMIM:116953'][0].semanticType).toEqual("Gene");
    })

    test("Test inputs with undefined semanticType and could not be mapped to any semantic type should return Irresolvable", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "undefined": ["OMIM1:116953"] });
        expect(res).toHaveProperty("OMIM1:116953");
        expect(res['OMIM1:116953'][0]).toBeInstanceOf(IrresolvableBioEntity);
        expect(res['OMIM1:116953'][0].semanticType).toEqual("undefined");
    })

    test("Test Irresolvable inputs should not overwrite the result of a valid input", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Gene": ["NCBIGene:1017"], "Disease": ["NCBIGene:1017"] });
        expect(res).toHaveProperty("NCBIGene:1017");
        expect(res["NCBIGene:1017"]).toHaveLength(2);
        expect(res['NCBIGene:1017'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['NCBIGene:1017'][0].primaryID).toEqual("NCBIGene:1017");
        expect(res['NCBIGene:1017'][0].label).toEqual("CDK2");
        expect(res['NCBIGene:1017'][1]).toBeInstanceOf(IrresolvableBioEntity);
    })

    // skip because new SRI-based ID resolver doesn't fetch node attributes
    test.skip("Test chemical attributes are correctly retrieved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "SmallMolecule": ["CHEMBL.COMPOUND:CHEMBL744"] });
        expect(res["CHEMBL.COMPOUND:CHEMBL744"][0].attributes.drugbank_taxonomy_class).toContain("Benzothiazoles");
        expect(res["CHEMBL.COMPOUND:CHEMBL744"][0].attributes.chembl_max_phase).toContain("4");
        expect(res["CHEMBL.COMPOUND:CHEMBL744"][0].attributes.chembl_molecule_type).toContain("Small molecule");
        expect(res["CHEMBL.COMPOUND:CHEMBL744"][0].attributes.drugbank_drug_category).toContain("Anticonvulsants");
        expect(res["CHEMBL.COMPOUND:CHEMBL744"][0].attributes.drugbank_groups).toContain("approved");
        expect(res["CHEMBL.COMPOUND:CHEMBL744"][0].attributes.drugbank_kingdom).toContain("Organic compounds");
        expect(res["CHEMBL.COMPOUND:CHEMBL744"][0].attributes.drugbank_superclass).toContain('Organoheterocyclic compounds');
        expect(res["CHEMBL.COMPOUND:CHEMBL744"][0].attributes.contraindications).toContain('Drug-induced hepatitis');
        expect(res["CHEMBL.COMPOUND:CHEMBL744"][0].attributes.contraindications).toContain('Drug-induced hepatitis');
        expect(res["CHEMBL.COMPOUND:CHEMBL744"][0].attributes.indications).toContain('Amyotrophic lateral sclerosis');
        expect(res["CHEMBL.COMPOUND:CHEMBL744"][0].attributes.mesh_pharmacology_class).toContain('Anticonvulsants');
        expect(res["CHEMBL.COMPOUND:CHEMBL744"][0].attributes.fda_epc_pharmacology_class).toContain('Benzothiazole');
    })

    test("Test variant attributes are correctly retrieved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "SequenceVariant": ["DBSNP:rs796065306"] });
        expect(res["DBSNP:rs796065306"][0].attributes.cadd_consequence).toContain("NON_SYNONYMOUS");
        expect(res["DBSNP:rs796065306"][0].attributes.cadd_variant_type).toContain("SNV");
        expect(res["DBSNP:rs796065306"][0].attributes.dbsnp_variant_type).toContain("snv");
        expect(res["DBSNP:rs796065306"][0].attributes.clinvar_clinical_significance).toContain("Pathogenic");
        expect(res["DBSNP:rs796065306"][0].attributes.sift_category).toContain("deleterious");
    })

    test("Test gene attributes are correctly retrieved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Gene": ["NCBIGene:1017"] });
        expect(res["NCBIGene:1017"][0].attributes.interpro).toContain("Protein kinase domain");
        expect(res["NCBIGene:1017"][0].attributes.type_of_gene).toContain("protein-coding");

    })

    test("Test pathway attributes are correctly retrieved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Pathway": ["WIKIPATHWAYS:WP151"] });
        expect(res["WIKIPATHWAYS:WP151"][0].attributes.num_of_participants).toContain("68");

    })

    test("Test pathway biocarta id are correctly resolved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Pathway": ["BIOCARTA:hbxpathway"] });
        expect(res["BIOCARTA:hbxpathway"][0]).toBeInstanceOf(ResolvableBioEntity);
    })

    test("Test SNOMEDCT id are correctly resolved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Disease": ["SNOMEDCT:49455004"] });
        console.log(res)
        expect(res["SNOMEDCT:49455004"][0]).toBeInstanceOf(ResolvableBioEntity);
    })

    test("Test NCIT id are correctly resolved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "Disease": ["NCIT:C116936"] });
        console.log(res)
        expect(res["NCIT:C116936"][0]).toBeInstanceOf(ResolvableBioEntity);
    })

    // update RHEA test to use MolecularActivity semantic type
    test("Test molecular activity ids can be resolved as RHEA ids", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "MolecularActivity": ["GO:0010176"] });
        expect(res["GO:0010176"][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res["GO:0010176"][0].dbIDs).toHaveProperty("RHEA");
        expect(res["GO:0010176"][0].dbIDs.RHEA).toContain("RHEA:37975")
    })

    // update RHEA test to use MolecularActivity semantic type
    test("Test RHEA ids can be correctly resolved", async () => {
        const resolver = new DefaultIDResolver();
        const res = await resolver.resolve({ "MolecularActivity": ["RHEA:37975"] });
        expect(res["RHEA:37975"][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res["RHEA:37975"][0].primaryID).toEqual("GO:0010176")
    })
})