import test from "node:test";
import assert from "node:assert/strict";
import {NexusRuntime} from "../web/runtime.js";

function fixture() {
  const c={vocab_size:256,width:4,layers:2,heads:2,context:8,experts:2,top_k:1};
  const tensors=[];
  const add=(name,shape,fill=0)=>{const length=shape.reduce((a,b)=>a*b,1);tensors.push({name,shape,values:new Float32Array(length).fill(fill)});};
  add("embedding.weight",[256,4]);add("final_norm.weight",[4],1);add("final_norm.bias",[4]);add("lm_head.weight",[256,4]);
  for(let i=0;i<c.layers;i++){
    const p="blocks."+i+".";
    for(const n of ["norm_ssm","norm_moe"]) {add(p+n+".weight",[4],1);add(p+n+".bias",[4]);}
    add(p+"selective_ssm.parameters.weight",[12,4]);add(p+"selective_ssm.parameters.bias",[12]);
    add(p+"selective_ssm.output.weight",[4,4]);add(p+"selective_ssm.output.bias",[4]);
    add(p+"selective_ssm.A_log",[4]);add(p+"selective_ssm.D",[4],1);
    add(p+"router.weight",[2,4]);add(p+"router.bias",[2]);
    if(i%2===1){add(p+"norm_attention.weight",[4],1);add(p+"norm_attention.bias",[4]);add(p+"qkv.weight",[12,4]);add(p+"qkv.bias",[12]);add(p+"attention_out.weight",[4,4]);add(p+"attention_out.bias",[4]);}
    for(let e=0;e<2;e++){const q=p+"experts."+e+".";add(q+"up.weight",[16,4]);add(q+"up.bias",[16]);add(q+"gate.weight",[16,4]);add(q+"gate.bias",[16]);add(q+"down.weight",[4,16]);add(q+"down.bias",[4]);}
  }
  let offset=0;const manifestT=[];const parts=[];
  for(const t of tensors){manifestT.push({name:t.name,shape:t.shape,offset,length:t.values.byteLength});parts.push(new Uint8Array(t.values.buffer));offset+=t.values.byteLength;}
  const bytes=new Uint8Array(offset);let at=0;for(const p of parts){bytes.set(p,at);at+=p.length;}
  const manifest={format:"NEXUS_WEB_V1",config:c,tensors:manifestT};
  const tokenizer="NEXUS_BPE_V1\n256\n";
  return {manifest,bytes:bytes.buffer,tokenizer};
}
test("loads a model, runs causal forward inference and generates tokens",()=>{
  const f=fixture(),rt=new NexusRuntime(f.manifest,f.bytes,f.tokenizer);
  assert.equal(rt.encode("hello").length,5);
  assert.equal(rt.decode(rt.encode("hello")),"hello");
  const logits=rt.logits([104,105]);
  assert.equal(logits.length,256);
  const generated=rt.generate("hi",2);
  assert.equal(generated.length,4);
});
