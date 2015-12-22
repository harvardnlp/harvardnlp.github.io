---
layout: page
title: Meetings
permalink: /meetings/
---


{% for term in site.data.meetings %}
<div class ="row">

<div style="text-align:center">
<h3 > {{term.term}} </h3>
</div>
</div>

<table class="table table-striped table-hover">
<tr><th> Date</th> <th> Paper</th> <th>Presenter </th></tr>
{% for paper in term.meetings %}
<tr><td> {{ paper.date }}  </td> <td><a href="paper.link"> {{paper.paper }}</a> </td><td> {{paper.presenter}} </td></tr>
{% endfor %}
</table>
{% endfor %}

