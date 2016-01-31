---
layout: page
title: Code
permalink: /code/
---

The following are research projects that have developed into open-source libraries:

<div class ="row"></div>

{% for project in site.data.code %}

<div class="panel panel-default">
<div class="panel-heading">
{{project.title}} 
</div>
<div class="panel-body">
<p>{{project.abstract}}</p>
<a href="{{project.link}}">GitHub</a>
</div>
</div>

{% endfor %}
