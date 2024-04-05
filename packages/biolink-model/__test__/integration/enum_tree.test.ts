import BioLinkEnumTree from '../../src/tree/enum_tree';
import QualifierObject from '../../src/object/qualifier_object';
import NodeNotFound from '../../src/exceptions/node_not_found'
import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import { BioLinkJSON } from '../../src/types/types'

describe("Test BioLink Enum Tree class", () => {
    describe("Test construct function", () => {
        let tree;
        let objs;

        beforeEach(() => {
            const file = fs.readFileSync(path.resolve(__dirname, '../../data/biolink.yaml'), { encoding: 'utf8' });
            const jsonObj = yaml.load(file) as BioLinkJSON;
            objs = jsonObj.enums;
            tree = new BioLinkEnumTree(objs);
        })

        test("Test all objects are corretly loaded", () => {
            tree.construct();
            expect(tree.objects).toHaveProperty("genetic_variant_form");
            expect(tree.objects.genetic_variant_form).toBeInstanceOf(QualifierObject);
            expect(tree.objects).toHaveProperty("reduction");
            expect(tree.objects).toHaveProperty("over_the_counter");
        })

        test("Test hierarchical order are correctly parsed", () => {
            tree.construct();
            expect(tree.objects.genetic_variant_form.children).toContain('polymorphic_form');
            expect(tree.objects.binding.children).toContain('inhibition');
            expect(tree.objects.inhibition.children).not.toContain('binding');
            expect(tree.objects.molecular_modification.children.sort()).toEqual([
                'acetylation',
                'acylation',
                'alkylation',
                'amination',
                'carbamoylation',
                'ethylation',
                'glutathionylation',
                'glycation',
                'glycosylation',
                'glucuronidation',
                'n_linked_glycosylation',
                'o_linked_glycosylation',
                'hydroxylation',
                'lipidation',
                'farnesylation',
                'geranoylation',
                'myristoylation',
                'palmitoylation',
                'prenylation',
                'methylation',
                'nitrosation',
                'nucleotidylation',
                'phosphorylation',
                'ribosylation',
                'ADP-ribosylation',
                'sulfation',
                'sumoylation',
                'ubiquitination',
                'oxidation',
                'reduction',
                'carboxylation',
            ].sort());
        })
    })

    describe("Test getDescendants function", () => {
        let tree;

        beforeEach(() => {
            const file = fs.readFileSync(path.resolve(__dirname, '../../data/biolink.yaml'), { encoding: 'utf8' });
            const jsonObj = yaml.load(file) as BioLinkJSON;
            const objs = jsonObj.enums;
            tree = new BioLinkEnumTree(objs);
            tree.construct();
        })

        test("Test multi-level inheritency is correctly parsed", () => {
            expect(tree.getDescendants("degradation")).toContain(tree.objects.cleavage);
            expect(tree.getDescendants("cleavage")).not.toContain(tree.objects.degradation);
        })

        test("Entity without descendants should return empty array", () => {
            expect(tree.getDescendants('cleavage')).toHaveLength(0);
            expect(tree.getDescendants('left')).toHaveLength(0);
        })

        test("Entity not in the tree should throw an error", () => {
            expect(() => {
                tree.getDescendants("Qualifier1");
            }).toThrowError(new NodeNotFound("The node you provide Qualifier1 is not in the tree."))
        })

    })

    describe("Test getAncestors function", () => {
        let tree;

        beforeEach(() => {
            const file = fs.readFileSync(path.resolve(__dirname, '../../data/biolink.yaml'), { encoding: 'utf8' });
            const jsonObj = yaml.load(file) as BioLinkJSON;
            const objs = jsonObj.enums;
            tree = new BioLinkEnumTree(objs);
            tree.construct();
        })

        test("Test multi-level inheritency is correctly parsed", () => {
            tree.construct();
            expect(tree.getAncestors("cleavage")).toContain(tree.objects.degradation);
            expect(tree.getAncestors("cleavage")).not.toContain(tree.objects.folding);
        })

        test("Entity without ancestors should return empty array", () => {
            expect(tree.getAncestors("folding")).toHaveLength(0);
        })

        test("Entity not in the tree should throw an error", () => {
            expect(() => {
                tree.getAncestors("Qualifier1");
            }).toThrowError(new NodeNotFound("The node you provide Qualifier1 is not in the tree."))
        })

    })

    describe("Test getPath function", () => {
        let tree;
        beforeEach(() => {
            const file = fs.readFileSync(path.resolve(__dirname, '../../data/biolink.yaml'), { encoding: 'utf8' });
            const jsonObj = yaml.load(file) as BioLinkJSON;
            const objs = jsonObj.enums;
            tree = new BioLinkEnumTree(objs);
            tree.construct();
        })

        // skipped as there are no existing paths to test this on anymore
        test.skip("Return all intermediates nodes between upstream and downstream if there're > 1 intermediates", () => {
            const res = tree.getPath("cleavage", "activity_or_abundance").map(item => item.name);
            expect(res).toEqual(["degradation", "abundance"]);
        })

        test("Return the intermediate nodes between upstream and downstream if there're only 1 intermediates", () => {
            const res = tree.getPath("synthesis", "activity_or_abundance").map(item => item.name);
            expect(res).toEqual(["abundance"]);
        })

        test("Return [] if upstream is direct parent of downstream", () => {
            const res = tree.getPath("abundance", "activity_or_abundance").map(item => item.name);
            expect(res).toEqual([]);
        })

        test("Return [] if downstream has no parent", () => {
            const res = tree.getPath("folding", "activity_or_abundance").map(item => item.name);
            expect(res).toEqual([]);
        })

        test("Downstream node not in the tree should throw an error", () => {
            expect(() => {
                tree.getPath("affects1", "affects2");
            }).toThrowError(new NodeNotFound("The node you provide affects1 is not in the tree."))
        })

        test("Upstream node not in the tree should throw an error", () => {
            expect(() => {
                tree.getPath("degradation", "affects2");
            }).toThrowError(new NodeNotFound("The node you provide affects2 is not in the tree."))
        })
    })
})
