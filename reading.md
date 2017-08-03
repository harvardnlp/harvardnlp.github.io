---
layout: page
title: Meetings
permalink: /meetings/
weight: 5
---


<ul>
<li>Time: Fall term Time TBD</li>
<li>Room: Maxwell Dworkin 221</li>
<li>Open to any interested party who reads the paper</li>
</ul>

{% for term in site.data.meetings %}
<div class ="row">

<div style="text-align:center">
<h3>{{term.term}}</h3>
</div>
</div>

<table class="table table-striped table-hover">
<tr>
	<th> Date</th> <th> Paper</th> <th>Presenter </th>
</tr>
{% for paper in term.meetings %}
	<tr>
		<td> {{ paper.date }}  </td>
		<td>
			{% if paper.cite %}
			<a href="{{paper.cite}}">{{paper.paper}}</a>
			{% else %}
			{{paper.paper}}
			{% endif %}
		</td>
		<td> {{paper.presenter}} </td>
	</tr>
{% endfor %}
</table>
{% endfor %}

<style>
#pubTable_filter{
    display:none;
}
</style>

<table id="pubTable" class="table table-hover"></table>
<script>
$(function(){
bibtexify("reading.bib", "pubTable", {"visualization":false});}
);
</script>
