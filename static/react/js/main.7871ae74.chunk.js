(this["webpackJsonpreact-fronend"]=this["webpackJsonpreact-fronend"]||[]).push([[0],{10:function(t,e,n){},11:function(t,e,n){"use strict";n.r(e);var c=n(1),i=n.n(c),s=n(0),o=n(3),a=n.n(o);n(9),n(10);var r=function(){return Object(s.jsxs)("div",{class:"main-container",children:[Object(s.jsx)("div",{class:"output-box",children:Object(s.jsx)("img",{id:"",alt:""})}),Object(s.jsxs)("div",{class:"wrapper",children:[Object(s.jsx)("img",{id:"input_image"}),Object(s.jsxs)("div",{class:"btn-container",children:[Object(s.jsxs)("label",{class:"custom-file-upload btn upload",children:[Object(s.jsx)("input",{type:"file",onChange:function(t){var e=new FileReader;e.onload=function(){document.getElementById("input_image").src=e.result,fetch("/result",{method:"POST",cache:"no-cache",headers:{"Content-type":"application/json"},body:JSON.stringify(e.result)}).then((function(t){return t.json()})).then((function(t){return console.log(t)})).catch((function(t){console.log(t)}))},e.readAsDataURL(t.target.files[0])}}),"Upload"]}),Object(s.jsx)("input",{type:"button",class:"btn fil",value:"Filter"})]})]})]})},l=function(t){t&&t instanceof Function&&n.e(3).then(n.bind(null,12)).then((function(e){var n=e.getCLS,c=e.getFID,i=e.getFCP,s=e.getLCP,o=e.getTTFB;n(t),c(t),i(t),s(t),o(t)}))};a.a.render(Object(s.jsx)(i.a.StrictMode,{children:Object(s.jsx)(r,{})}),document.getElementById("root")),l()},9:function(t,e,n){}},[[11,1,2]]]);
//# sourceMappingURL=main.7871ae74.chunk.js.map