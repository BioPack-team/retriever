import BioLinkClassTree from '../../src/tree/class_tree';
import EntityObject from '../../src/object/entity_object';
import NodeNotFound from '../../src/exceptions/node_not_found'
import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import { BioLinkJSON } from '../../src/types/types'

describe("Test BioLink Class Tree class", () => {
    describe("Test construct function", () => {
        let tree;
        let objs;

        beforeEach(() => {
            const file = fs.readFileSync(path.resolve(__dirname, '../../data/biolink.yaml'), { encoding: 'utf8' });
            const jsonObj = yaml.load(file) as BioLinkJSON;
            objs = jsonObj.classes;
            tree = new BioLinkClassTree(objs);
        })

        test("Test all objects are corretly loaded", () => {
            tree.construct();
            expect(tree.objects).toHaveProperty("Gene");
            expect(tree.objects.Gene).toBeInstanceOf(EntityObject)
            expect(tree.objects).toHaveProperty("SmallMolecule");
            expect(tree.objects).toHaveProperty("OntologyClass");
            expect(Object.keys(tree.objects)).toHaveLength(Object.keys(objs).length);
        })

        test("Test hierarchical order are correctly parsed", () => {
            tree.construct();
            expect(tree.objects.MolecularEntity.children).toEqual(["SmallMolecule", "NucleicAcidEntity"]);
            expect(tree.objects.MolecularEntity.children).not.toContain('Gene')
            expect(tree.objects.Gene.children).toHaveLength(0);
        })
    })

    describe("Test getDescendants function", () => {
        let tree;

        beforeEach(() => {
            const file = fs.readFileSync(path.resolve(__dirname, '../../data/biolink.yaml'), { encoding: 'utf8' });
            const jsonObj = yaml.load(file) as BioLinkJSON;
            const objs = jsonObj.classes;
            tree = new BioLinkClassTree(objs);
            tree.construct();
        })

        test("Test multi-level inheritency is correctly parsed", () => {
            expect(tree.getDescendants("MolecularEntity")).toEqual(
                [
                    tree.objects.SmallMolecule,
                    tree.objects.NucleicAcidEntity,
                    // tree.objects.Exon,
                    // tree.objects.Transcript,
                    // tree.objects.RnaProduct,
                    // tree.objects.RnaProductIsoform,
                    // tree.objects.NoncodingRnaProduct,
                    // tree.objects.MicroRna,
                    // tree.objects.SiRna,
                    // tree.objects.CodingSequence
                ]
            );
            expect(tree.objects.NamedThing).not.toBeUndefined;
            expect(tree.getDescendants("MolecularEntity")).not.toContain(tree.objects.NamedThing);
        })

        test("Entity without descendants should return empty array", () => {
            expect(tree.getDescendants("Gene")).toHaveLength(0);
            expect(tree.getDescendants('ProteinIsoform')).toHaveLength(0);
        })

        test("Entity not in the tree should throw an error", () => {
            expect(() => {
                tree.getDescendants("Gene1");
            }).toThrowError(new NodeNotFound("The node you provide Gene1 is not in the tree."))
        })

    })

    describe("Test getAncestors function", () => {
        let tree;

        beforeEach(() => {
            const file = fs.readFileSync(path.resolve(__dirname, '../../data/biolink.yaml'), { encoding: 'utf8' });
            const jsonObj = yaml.load(file) as BioLinkJSON;
            const objs = jsonObj.classes;
            tree = new BioLinkClassTree(objs);
            tree.construct();
        })

        test("Test multi-level inheritency is correctly parsed", () => {
            tree.construct();
            expect(tree.getAncestors("Gene")).toEqual(
                [
                    tree.objects.BiologicalEntity,
                    tree.objects.NamedThing,
                    tree.objects.Entity
                ]
            );
            expect(tree.getAncestors("Gene")).toContain(tree.objects.NamedThing);
            expect(tree.getAncestors("Gene")).not.toContain(tree.objects.Protein);
        })

        test("Entity without ancestors should return empty array", () => {
            expect(tree.getAncestors("Entity")).toHaveLength(0);
            expect(tree.getAncestors('Annotation')).toHaveLength(0);
        })

        test("Entity not in the tree should throw an error", () => {
            expect(() => {
                tree.getAncestors("Gene1");
            }).toThrowError(new NodeNotFound("The node you provide Gene1 is not in the tree."))
        })

    })

    describe("Test getPath function", () => {
        let tree;
        beforeEach(() => {
            const file = fs.readFileSync(path.resolve(__dirname, '../../data/biolink.yaml'), { encoding: 'utf8' });
            const jsonObj = yaml.load(file) as BioLinkJSON;
            const objs = jsonObj.classes;
            tree = new BioLinkClassTree(objs);
            tree.construct();
        })

        test("Return all intermediates nodes between upstream and downstream if there're > 1 intermediates", () => {
            const res = tree.getPath("Gene", "NamedThing").map(item => item.name);
            expect(res).toEqual(["BiologicalEntity"]);
        })

        test("Return the intermediate nodes between upstream and downstream if there're only 1 intermediates", () => {
            const res = tree.getPath("Gene", "MolecularEntity").map(item => item.name);
            expect(res).toEqual(["BiologicalEntity", "NamedThing", "Entity"]);
        })

        test("Return [] if upstream is direct parent of downstream", () => {
            const res = tree.getPath("Gene", "GenomicEntity").map(item => item.name);
            expect(res).toEqual(["BiologicalEntity", "NamedThing", "Entity"]);
        })

        test("Return [] if downstream has no parent", () => {
            const res = tree.getPath("Entity", "GenomicEntity").map(item => item.name);
            expect(res).toEqual([]);
        })

        test("Downstream node not in the tree should throw an error", () => {
            expect(() => {
                tree.getPath("Gene1", "Gene2");
            }).toThrowError(new NodeNotFound("The node you provide Gene1 is not in the tree."))
        })

        test("Upstream node not in the tree should throw an error", () => {
            expect(() => {
                tree.getPath("Gene", "Gene2");
            }).toThrowError(new NodeNotFound("The node you provide Gene2 is not in the tree."))
        })
    })
})
