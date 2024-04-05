import generateInvalid from '../../src/fake';
import { IrresolvableBioEntity } from '../../src/bioentity/irresolvable_bioentity'

describe("Test generateInvalid function", () => {
    test("invalid inputs should be generated", () => {
        const input = {
            "Gene": ["NCBIGene:1017", "NCBIGene:1018"]
        }
        const res = generateInvalid(input);
        expect(res).toHaveProperty("NCBIGene:1017");
        expect(res).toHaveProperty("NCBIGene:1018");
        expect(res["NCBIGene:1017"][0]).toBeInstanceOf(IrresolvableBioEntity);
    })
})