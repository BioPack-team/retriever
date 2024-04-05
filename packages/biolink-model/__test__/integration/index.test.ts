import { BioLink } from '../../src/index';
import BioLinkClassTree from '../../src/tree/class_tree';
import BioLinkSlotTree from '../../src/tree/slot_tree';

describe("Test BioLink module", () => {
    describe("Test load function", () => {
        test("When source is unspecified, should return default spec", async () => {
            const biolink = new BioLink();
            await biolink.load();
            expect(biolink.biolinkJSON).toHaveProperty("classes");
        })
    })

    describe("Test loadSync function", () => {
        test("When source is unspecified, should return default spec", () => {
            const biolink = new BioLink();
            biolink.loadSync();
            expect(biolink.biolinkJSON).toHaveProperty("classes");
        })
    })

    describe("Test classTree getter method", () => {
        test("BioLink Class Tree Object is correctly retrieved in async mode", async () => {
            const biolink = new BioLink();
            await biolink.load();
            expect(biolink.classTree).toBeInstanceOf(BioLinkClassTree);
            expect(biolink.classTree.objects).toHaveProperty("Gene");
        })

        test("BioLink Class Tree Object is correctly retrieved in sync mode", () => {
            const biolink = new BioLink();
            biolink.loadSync();
            expect(biolink.classTree).toBeInstanceOf(BioLinkClassTree);
            expect(biolink.classTree.objects).toHaveProperty("Gene");
        })
    })

    describe("Test slotTree getter method", () => {
        test("BioLink Slot Tree Object is correctly retrieved in async mode", async () => {
            const biolink = new BioLink();
            await biolink.load();
            expect(biolink.slotTree).toBeInstanceOf(BioLinkSlotTree);
            expect(biolink.slotTree.objects).toHaveProperty("related_to");
        })

        test("BioLink Slot Tree Object is correctly retrieved in sync mode", () => {
            const biolink = new BioLink();
            biolink.loadSync();
            expect(biolink.slotTree).toBeInstanceOf(BioLinkSlotTree);
            expect(biolink.slotTree.objects).toHaveProperty("related_to");
        })
    })
})