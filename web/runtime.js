/* NEXUS browser inference runtime. CPU reference backend; no external dependencies. */
export class NexusRuntime {
  constructor(manifest, weightBuffer, tokenizerText) {
    if (manifest.format !== "NEXUS_WEB_V1") throw new Error("Unsupported NEXUS model format");
    this.cfg = manifest.config;
    this.w = new Map();
    const bytes = weightBuffer instanceof ArrayBuffer ? weightBuffer : weightBuffer.buffer;
    const base = weightBuffer instanceof ArrayBuffer ? 0 : weightBuffer.byteOffset;
    for (const t of manifest.tensors) {
      const elementCount = t.shape.reduce((a,b)=>a*b,1);
      if (t.offset % 4 || t.length % 4 || t.length !== elementCount * 4 || base + t.offset + t.length > bytes.byteLength)
        throw new Error("Invalid tensor byte range or shape in model file: " + t.name);
      this.w.set(t.name, new Float32Array(bytes, base + t.offset, elementCount));
    }
    this.tokens = [];
    this.merges = [];
    const lines = tokenizerText.trim().split(/\r?\n/);
    if (lines[0] !== "NEXUS_BPE_V1") throw new Error("Unsupported tokenizer format");
    const count = Number(lines[1]);
    for (let i=0;i<256;i++) this.tokens.push(Uint8Array.of(i));
    for (let i=2;i<lines.length && this.merges.length<count-256;i++) {
      const [a,b] = lines[i].trim().split(/\s+/).map(Number);
      if (!Number.isInteger(a)||!Number.isInteger(b)||a<0||b<0||a>=this.tokens.length||b>=this.tokens.length) throw new Error("Invalid tokenizer merge table");
      this.merges.push([a,b]);
      const joined = new Uint8Array(this.tokens[a].length+this.tokens[b].length);
      joined.set(this.tokens[a]); joined.set(this.tokens[b],this.tokens[a].length);
      this.tokens.push(joined);
    }
    if (this.tokens.length !== count) throw new Error("Tokenizer vocabulary does not match its header");
    this.ranks = new Map(this.merges.map((p,i)=>[p[0]+":"+p[1],i]));
    this.validateWeights();
  }
  validateWeights() {
    const c=this.cfg, required=["embedding.weight","final_norm.weight","final_norm.bias","lm_head.weight","lm_head.bias"];
    for(const n of required) if(!this.w.has(n)) throw new Error("Model is missing tensor "+n);
    if(c.width<1||c.layers<1||c.heads<1||c.width%c.heads) throw new Error("Invalid model configuration");
    for(let i=0;i<c.layers;i++) {
      const p="blocks."+i+".";
      for(const n of ["norm_ssm.weight","norm_ssm.bias","selective_ssm.parameters.weight","selective_ssm.parameters.bias","selective_ssm.output.weight","selective_ssm.output.bias","selective_ssm.A_log","selective_ssm.D","norm_moe.weight","norm_moe.bias","router.weight","router.bias"]) if(!this.w.has(p+n)) throw new Error("Missing tensor "+p+n);
      if(i%2===1) for(const n of ["norm_attention.weight","norm_attention.bias","qkv.weight","qkv.bias","attention_out.weight","attention_out.bias"]) if(!this.w.has(p+n)) throw new Error("Missing tensor "+p+n);
      for(let e=0;e<c.experts;e++) for(const n of ["up.weight","up.bias","gate.weight","gate.bias","down.weight","down.bias"]) if(!this.w.has(p+"experts."+e+"."+n)) throw new Error("Missing expert tensor "+p+"experts."+e+"."+n);
    }
  }
  param(name) { const x=this.w.get(name); if(!x) throw new Error("Missing parameter "+name); return x; }
  encode(text) {
    const raw=new TextEncoder().encode(text), n=raw.length;
    if(!n) return [];
    const v=Array.from(raw), prev=Array.from({length:n},(_,i)=>i-1), next=Array.from({length:n},(_,i)=>i+1<n?i+1:-1), alive=new Uint8Array(n).fill(1);
    while(true) {
      let best=Infinity, at=-1;
      for(let i=0;i<n;i++) if(alive[i]&&next[i]>=0) { const rank=this.ranks.get(v[i]+":"+v[next[i]]); if(rank!==undefined&&rank<best){best=rank;at=i;} }
      if(at<0) break;
      const right=next[at], after=next[right]; v[at]=256+best; next[at]=after; alive[right]=0;
      if(after>=0) prev[after]=at;
    }
    const out=[]; for(let i=0;i<n;i++) if(alive[i]) out.push(v[i]); return out;
  }
  decode(ids) {
    let size=0; const chunks=[];
    for(const id of ids){const t=this.tokens[id];if(!t)throw new Error("Invalid generated token id "+id);chunks.push(t);size+=t.length;}
    const all=new Uint8Array(size);let at=0;for(const t of chunks){all.set(t,at);at+=t.length;}
    return new TextDecoder("utf-8",{fatal:false}).decode(all);
  }
  linear(x, rows, input, output, prefix) {
    const W=this.param(prefix+".weight"), B=this.param(prefix+".bias"), y=new Float32Array(rows*output);
    for(let r=0;r<rows;r++) for(let o=0;o<output;o++){let s=B[o];const wi=o*input, xi=r*input;for(let k=0;k<input;k++)s+=W[wi+k]*x[xi+k];y[r*output+o]=s;}
    return y;
  }
  norm(x, rows, width, prefix) {
    const g=this.param(prefix+".weight"),b=this.param(prefix+".bias"),y=new Float32Array(x.length);
    for(let r=0;r<rows;r++){const off=r*width;let mean=0;for(let j=0;j<width;j++)mean+=x[off+j];mean/=width;let variance=0;for(let j=0;j<width;j++){const d=x[off+j]-mean;variance+=d*d;}variance/=width;const inv=1/Math.sqrt(variance+1e-5);for(let j=0;j<width;j++)y[off+j]=(x[off+j]-mean)*inv*g[j]+b[j];}
    return y;
  }
  ssm(x,T,p) {
    const d=this.cfg.width, proj=this.linear(x,T,d,d*3,p+"selective_ssm.parameters"), A=this.param(p+"selective_ssm.A_log"), D=this.param(p+"selective_ssm.D"), state=new Float32Array(d), z=new Float32Array(T*d);
    for(let t=0;t<T;t++) for(let j=0;j<d;j++){
      const raw=proj[t*3*d+j], delta=(raw>20?raw:Math.log1p(Math.exp(Math.max(-30,raw))))+1e-4;
      const a=Math.exp(-delta*Math.exp(A[j])), gate=Math.tanh(proj[t*3*d+d+j]);
      state[j]=a*state[j]+(1-a)*gate*x[t*d+j];
      z[t*d+j]=Math.tanh(proj[t*3*d+2*d+j])*state[j]+D[j]*x[t*d+j];
    }
    return this.linear(z,T,d,d,p+"selective_ssm.output");
  }
  attention(x,T,p) {
    const c=this.cfg,d=c.width,h=c.heads,hd=d/h,qkv=this.linear(x,T,d,d*3,p+"qkv"),out=new Float32Array(T*d),scale=1/Math.sqrt(hd);
    for(let head=0;head<h;head++) for(let t=0;t<T;t++){
      const scores=new Float32Array(t+1);let max=-Infinity;
      for(let s=0;s<=t;s++){let dot=0;for(let j=0;j<hd;j++)dot+=qkv[t*3*d+head*hd+j]*qkv[s*3*d+d+head*hd+j];scores[s]=dot*scale;if(scores[s]>max)max=scores[s];}
      let denom=0;for(let s=0;s<=t;s++){scores[s]=Math.exp(scores[s]-max);denom+=scores[s];}
      for(let j=0;j<hd;j++){let value=0;for(let s=0;s<=t;s++)value+=(scores[s]/denom)*qkv[s*3*d+2*d+head*hd+j];out[t*d+head*hd+j]=value;}
    }
    return this.linear(out,T,d,d,p+"attention_out");
  }
  swiglu(x,T,d,p) {
    const up=this.linear(x,T,d,d*4,p+".up"),gate=this.linear(x,T,d,d*4,p+".gate"),hidden=new Float32Array(up.length);
    for(let i=0;i<hidden.length;i++){const g=gate[i],s=1/(1+Math.exp(-Math.max(-30,Math.min(30,g))));hidden[i]=(g*s)*up[i];}
    return this.linear(hidden,T,d*4,d,p+".down");
  }
  block(x,T,i) {
    const c=this.cfg,d=c.width,p="blocks."+i+".";
    const s=this.ssm(this.norm(x,T,d,p+"norm_ssm"),T,p);
    for(let k=0;k<x.length;k++)x[k]+=s[k];
    if(i%2===1){const a=this.attention(this.norm(x,T,d,p+"norm_attention"),T,p);for(let k=0;k<x.length;k++)x[k]+=a[k];}
    const n=this.norm(x,T,d,p+"norm_moe"), logits=this.linear(n,T,d,c.experts,p+"router"), mixed=new Float32Array(x.length);
    for(let t=0;t<T;t++){
      const order=Array.from({length:c.experts},(_,e)=>e).sort((a,b)=>logits[t*c.experts+b]-logits[t*c.experts+a]).slice(0,c.top_k);
      let max=-Infinity;for(const e of order)max=Math.max(max,logits[t*c.experts+e]);let sum=0;const weights=order.map(e=>{const v=Math.exp(logits[t*c.experts+e]-max);sum+=v;return v;});
      for(let q=0;q<order.length;q++){const e=order[q],w=weights[q]/sum,ep=p+"experts."+e+".",one=n.slice(t*d,(t+1)*d),up=this.linear(one,1,d,d*4,ep+"up"),gate=this.linear(one,1,d,d*4,ep+"gate"),hidden=new Float32Array(d*4);for(let j=0;j<hidden.length;j++){const g=gate[j];hidden[j]=(g/(1+Math.exp(-Math.max(-30,Math.min(30,g)))))*up[j];}const result=this.linear(hidden,1,d*4,d,ep+"down");for(let j=0;j<d;j++)mixed[t*d+j]+=result[j]*w;}
    }
    for(let k=0;k<x.length;k++)x[k]+=mixed[k];return x;
  }
  logits(ids) {
    const c=this.cfg,d=c.width,T=ids.length;if(T<1||T>c.context)throw new Error("Prompt exceeds model context ("+c.context+" tokens)");
    const emb=this.param("embedding.weight"),x=new Float32Array(T*d);
    for(let t=0;t<T;t++){const id=ids[t];if(id<0||id>=c.vocab_size)throw new Error("Token id outside vocabulary");x.set(emb.subarray(id*d,(id+1)*d),t*d);}
    for(let i=0;i<c.layers;i++)this.block(x,T,i);
    const n=this.norm(x,T,d,"final_norm"),W=this.param("lm_head.weight"),B=this.param("lm_head.bias"),last=n.subarray((T-1)*d,T*d),out=new Float32Array(c.vocab_size);
    for(let v=0;v<c.vocab_size;v++){let s=B[v];for(let j=0;j<d;j++)s+=W[v*d+j]*last[j];out[v]=s;}return out;
  }
  async generate(prompt, count=64, onToken=()=>{}) {
    const ids=this.encode(prompt);if(!ids.length)throw new Error("Enter a prompt first");
    const generated=[...ids];
    for(let step=0;step<count;step++){
      const context=generated.slice(-this.cfg.context),scores=this.logits(context);
      const top=Array.from({length:scores.length},(_,i)=>i).sort((a,b)=>scores[b]-scores[a]).slice(0,Math.min(40,scores.length));
      const max=scores[top[0]],probs=top.map(i=>Math.exp((scores[i]-max)/0.8)),total=probs.reduce((a,b)=>a+b,0);
      let r=Math.random()*total,chosen=top[top.length-1];for(let i=0;i<top.length;i++){r-=probs[i];if(r<=0){chosen=top[i];break;}}
      generated.push(chosen);onToken(this.decode([chosen]),step+1);
      await new Promise(resolve=>setTimeout(resolve,0));
    }
    return this.decode(generated);
  }
}
