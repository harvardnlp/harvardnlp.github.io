---
layout: page
title: About
permalink: /about/
---
<div class ="row">
{% for member in site.data.members %}
<div class="col-md-4"><div><img class="img-thumbnail"></div>
<span class="lead">{{member.name}}</span><p>{{member.title}}</p> </div>
{% endfor %}
</div>
