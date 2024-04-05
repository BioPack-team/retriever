import BaseTransformer from "./transformer";

export default class EBIProteinTransformer extends BaseTransformer {
  wrap(res) {
    const new_comments = [];
    if (res && "comments" in res) {
      res.comments.map(comment => {
        if ("reaction" in comment) {
          comment.reaction.dbReferences = comment.reaction.dbReferences.filter(item => item.type === "Rhea");
        }
        new_comments.push(comment);
      });
    }
    res.comments = new_comments;
    return res;
  }
}
