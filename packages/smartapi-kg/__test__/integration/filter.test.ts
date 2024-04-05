import { ft } from '../../src/filter';
import { SmartAPIKGOperationObject } from '../../src/parser/types'

describe("Test filter function", () => {
    const ops: SmartAPIKGOperationObject[] = [
        {
            association: {
                input_type: "Gene",
                input_id: "NCBIGENE",
                output_type: "Disease",
                output_id: "MONDO",
                predicate: "PRED1",
                source: "SOURCE1",
                'x-translator': {
                    component: 'KP'
                }
            }
        },
        {
            association: {
                input_type: "SequenceVariant",
                input_id: "NCBIGENE",
                output_type: "SmallMolecule",
                output_id: "CHEBI",
                predicate: "PRED2",
                source: "SOURCE2",
                api_name: "API1",
                'x-translator': {
                    component: 'KP'
                }
            }
        },
        {
            association: {
                input_type: "Gene",
                input_id: "NCBIGENE",
                output_type: "SmallMolecule",
                output_id: "CHEBI",
                predicate: "PRED3",
                source: "SOURCE3",
                'x-translator': {
                    component: 'KP'
                }
            }
        },
        {
            association: {
                input_type: "Gene",
                input_id: "NCBIGENE",
                output_type: "Disease",
                output_id: "MONDO",
                predicate: "PRED2",
                source: "SOURCE3",
                api_name: "API1",
                'x-translator': {
                    component: 'ARA'
                }
            }
        },
    ]
    test("Return only ops with input type gene when specifying input type as gene in string", () => {
        const res = ft(ops, { input_type: "Gene" });
        expect(res).toHaveLength(ops.filter(op => op.association.input_type === "Gene").length);
        expect(res[0].association.source).toEqual("SOURCE1");
    })

    test("Return only KP ops", () => {
        const res = ft(ops, { component: "KP" });
        expect(res).toHaveLength(ops.filter(op => op.association['x-translator'].component === "KP").length);
        expect(res[0].association.source).toEqual("SOURCE1");
    })

    test("Return only ops with input type gene when specifying input type as gene in array", () => {
        const res = ft(ops, { input_type: ["Gene"] });
        expect(res).toHaveLength(ops.filter(op => op.association.input_type === "Gene").length);
        expect(res[0].association.source).toEqual("SOURCE1");
    })

    test("Return only ops with output type gene when specifying output type as SmallMolecule in string", () => {
        const res = ft(ops, { output_type: "SmallMolecule" });
        expect(res).toHaveLength(ops.filter(op => op.association.output_type === "SmallMolecule").length);
        expect(res[0].association.source).toEqual("SOURCE2");
    })

    test("Return only ops with predicate PRED2 when specifying predicate as PRED2 in string", () => {
        const res = ft(ops, { predicate: "PRED2" });
        expect(res).toHaveLength(ops.filter(op => op.association.predicate === "PRED2").length);
        expect(res[0].association.source).toEqual("SOURCE2");
    })

    test("Return only ops with input type Gene and output type Disease when specifying input as Gene and output as Disease", () => {
        const res = ft(ops, { input_type: 'Gene', output_type: "Disease" });
        expect(res).toHaveLength(ops.filter(op => op.association.input_type === 'Gene' && op.association.output_type === "Disease").length);
        expect(res[0].association.source).toEqual("SOURCE1");
    })

    test("Return only ops with input type Gene and output type Disease/SmallMolecule when specifying input as Gene and output as Disease/SmallMolecule", () => {
        const res = ft(ops, { input_type: 'Gene', output_type: ["Disease", "SmallMolecule"] });
        expect(res).toHaveLength(ops.filter(op => op.association.input_type === 'Gene' && ["Disease", "SmallMolecule"].includes(op.association.output_type)).length);
        expect(res[0].association.source).toEqual("SOURCE1");
    })

    test("Return empty array when couldn't find a match", () => {
        const res = ft(ops, { input_type: 'Gene1', output_type: ["Disease", "SmallMolecule"] });
        expect(res).toHaveLength(0);
    })
})
